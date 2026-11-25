//! One HTTP request to a model server or to Corvus' own MCP server, plain or
//! over TLS, that a stop request ends within a fraction of a second.
//!
//! A model that thinks for a minute sends nothing for a minute, so a read
//! cannot simply block: the socket reads in short slices and this code asks
//! after each one whether the operator pressed Stop, the window is closing, or
//! the time allowed is up.

use std::io::{self, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use corvus_api::net::http::read_response;

use super::model::Endpoint;

/// The most of an answer that is read into memory.
const BODY_MAX: u64 = 8 * 1024 * 1024;
const SLICE: Duration = Duration::from_millis(250);
const CONNECT: Duration = Duration::from_secs(3);

/// Whether the work in progress should end. Cloned into every request of a turn.
#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// What came back.
#[derive(Debug)]
pub struct Answer {
    pub status: u16,
    /// Names in lower case.
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Answer {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }
}

trait Io: Read + Write + Send {}
impl<T: Read + Write + Send> Io for T {}

/// A stream whose reads wait in slices and give up on `cancel` or the deadline.
struct Patient {
    inner: Box<dyn Io>,
    cancel: Cancel,
    deadline: Instant,
}

impl Read for Patient {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.cancel.is_set() {
                // Not `Interrupted`: the readers of `std` try that one again, for ever.
                return Err(io::Error::new(io::ErrorKind::ConnectionAborted, "stopped"));
            }
            if Instant::now() >= self.deadline {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "no answer in time"));
            }
            match self.inner.read(buf) {
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => continue,
                other => return other,
            }
        }
    }
}

impl Write for Patient {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn tls_config() -> Result<Arc<rustls::ClientConfig>, String> {
    static CONFIG: OnceLock<Result<Arc<rustls::ClientConfig>, String>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let mut roots = rustls::RootCertStore::empty();
            roots.add_parsable_certificates(rustls_native_certs::load_native_certs().certs);
            if roots.is_empty() {
                return Err("This computer has no trusted certificates, so https cannot be used.".into());
            }
            let provider = Arc::new(rustls::crypto::ring::default_provider());
            let config = rustls::ClientConfig::builder_with_provider(provider)
                .with_safe_default_protocol_versions()
                .map_err(|e| e.to_string())?
                .with_root_certificates(roots)
                .with_no_client_auth();
            Ok(Arc::new(config))
        })
        .clone()
}

fn connect(ep: &Endpoint, cancel: &Cancel) -> Result<TcpStream, String> {
    let mut last = format!("{} did not resolve.", ep.host);
    let addrs = (ep.host.as_str(), ep.port)
        .to_socket_addrs()
        .map_err(|e| format!("{}: {e}", ep.label()))?;
    for addr in addrs {
        if cancel.is_set() {
            return Err("Stopped.".into());
        }
        match TcpStream::connect_timeout(&addr, CONNECT) {
            Ok(s) => return Ok(s),
            Err(e) => last = format!("{}: {e}", ep.label()),
        }
    }
    Err(last)
}

fn read_body(mut body: impl Read) -> io::Result<String> {
    let mut bytes = Vec::new();
    match (&mut body).take(BODY_MAX).read_to_end(&mut bytes) {
        Ok(_) => {}
        // A TLS server that closes without saying so after its last byte.
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof && !bytes.is_empty() => {}
        Err(e) => return Err(e),
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Sends one request and reads the whole answer, which is `wait` at most.
pub fn call(
    ep: &Endpoint,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&str>,
    cancel: &Cancel,
    wait: Duration,
) -> Result<Answer, String> {
    let fail = |e: io::Error| match e.kind() {
        _ if cancel.is_set() => "Stopped.".to_string(),
        io::ErrorKind::TimedOut => format!("{} did not answer in time.", ep.label()),
        _ => format!("{}: {e}", ep.label()),
    };
    if headers
        .iter()
        .any(|(k, v)| k.contains(['\r', '\n']) || v.contains(['\r', '\n']))
    {
        return Err("A header holds a line break.".into());
    }
    let tcp = connect(ep, cancel)?;
    tcp.set_read_timeout(Some(SLICE)).map_err(fail)?;
    tcp.set_write_timeout(Some(Duration::from_secs(30))).map_err(fail)?;
    let _ = tcp.set_nodelay(true);
    let inner: Box<dyn Io> = if ep.tls {
        let name = rustls::pki_types::ServerName::try_from(ep.host.clone())
            .map_err(|_| format!("{}: not a valid host name.", ep.host))?;
        let conn = rustls::ClientConnection::new(tls_config()?, name).map_err(|e| e.to_string())?;
        Box::new(rustls::StreamOwned::new(conn, tcp))
    } else {
        Box::new(tcp)
    };
    let mut stream = Patient {
        inner,
        cancel: cancel.clone(),
        deadline: Instant::now() + wait,
    };
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nUser-Agent: corvus-gcs-ai-chat\r\n",
        ep.host_header()
    );
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(b) = body {
        head.push_str(&format!("Content-Length: {}\r\n", b.len()));
    }
    head.push_str("\r\n");
    head.push_str(body.unwrap_or(""));
    stream.write_all(head.as_bytes()).map_err(fail)?;
    stream.flush().map_err(fail)?;
    let response = read_response(BufReader::new(stream)).map_err(fail)?;
    let (status, headers) = (response.status, response.headers.clone());
    let body = read_body(response.body).map_err(fail)?;
    Ok(Answer { status, headers, body })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn whole(request: &[u8]) -> bool {
        let text = String::from_utf8_lossy(request);
        let Some((head, body)) = text.split_once("\r\n\r\n") else {
            return false;
        };
        let want = head
            .lines()
            .find_map(|l| l.strip_prefix("Content-Length: ")?.parse().ok())
            .unwrap_or(0);
        body.len() >= want
    }

    fn serve(reply: &'static [u8], hold: Duration) -> Endpoint {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                // The whole request: closing on unread bytes resets the connection, and
                // on Windows the client then loses the answer.
                let mut got = Vec::new();
                let mut buf = [0u8; 4096];
                while !whole(&got) {
                    match s.read(&mut buf) {
                        Ok(n) if n > 0 => got.extend_from_slice(&buf[..n]),
                        _ => return,
                    }
                }
                std::thread::sleep(hold);
                let _ = s.write_all(reply);
            }
        });
        Endpoint::parse(&format!("http://127.0.0.1:{port}")).unwrap()
    }

    #[test]
    fn a_json_answer_is_read_with_its_headers() {
        let ep = serve(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: abc\r\nContent-Length: 8\r\n\r\n{\"a\": 1}",
            Duration::ZERO,
        );
        let a = call(
            &ep,
            "POST",
            "/x",
            &[("Content-Type", "application/json")],
            Some("{}"),
            &Cancel::default(),
            Duration::from_secs(5),
        )
        .unwrap();
        assert_eq!((a.status, a.body.as_str()), (200, "{\"a\": 1}"));
        assert_eq!(a.header("mcp-session-id"), Some("abc"));
    }

    #[test]
    fn a_chunked_answer_is_read() {
        let ep = serve(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n",
            Duration::ZERO,
        );
        let a = call(&ep, "GET", "/", &[], None, &Cancel::default(), Duration::from_secs(5)).unwrap();
        assert_eq!(a.body, "abcde");
    }

    #[test]
    fn stop_ends_a_request_the_server_never_answers() {
        let ep = serve(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n", Duration::from_secs(30));
        let cancel = Cancel::default();
        let c = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(400));
            c.cancel();
        });
        let t = Instant::now();
        let e = call(&ep, "GET", "/", &[], None, &cancel, Duration::from_secs(60)).unwrap_err();
        assert_eq!(e, "Stopped.");
        assert!(t.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn a_slow_server_ends_at_the_time_allowed() {
        let ep = serve(b"", Duration::from_secs(30));
        let e = call(
            &ep,
            "GET",
            "/",
            &[],
            None,
            &Cancel::default(),
            Duration::from_millis(600),
        )
        .unwrap_err();
        assert!(e.contains("did not answer in time"), "{e}");
    }

    #[test]
    fn nothing_listening_is_an_error_naming_the_address() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        drop(l);
        let ep = Endpoint::parse(&format!("http://127.0.0.1:{port}")).unwrap();
        let e = call(&ep, "GET", "/", &[], None, &Cancel::default(), Duration::from_secs(2)).unwrap_err();
        assert!(e.contains(&port.to_string()), "{e}");
    }

    #[test]
    fn a_header_with_a_line_break_is_refused() {
        let ep = Endpoint::parse("http://127.0.0.1:9").unwrap();
        let e = call(
            &ep,
            "GET",
            "/",
            &[("X", "a\r\nb")],
            None,
            &Cancel::default(),
            Duration::from_secs(2),
        )
        .unwrap_err();
        assert!(e.contains("line break"));
    }
}
