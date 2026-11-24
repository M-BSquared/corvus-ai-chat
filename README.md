# AI Chat for Corvus GCS

A plugin for [Corvus GCS](https://github.com/M-BSquared/CorvusGCS), the
offline ground control station for PX4 and ArduPilot.

A chat in a tab of its own with a language model you choose, and the model can
use the whole station: it reads the aircraft and works the tools of Corvus'
MCP server, the same ones an AI client like Claude Desktop gets.

- **Any model server.** The OpenAI kind (chat completions: Ollama, LM Studio,
  llama.cpp, vLLM, OpenAI) or the Anthropic kind (messages: Claude and servers
  that copy it). On this computer, on the local network or hosted. A model on
  this computer works with no internet, which is the point.
- **Set up in Settings.** The gear of AI Chat under Settings, Plugins holds the
  address, the model, the key, the instructions the model is given, the reply
  length and how long to wait.
- **Check.** The round arrow at the top looks at the station's tools and lists the
  models the server offers.
- **Stop.** Ends a turn within a fraction of a second, also while the model is
  still thinking.

## The station's tools, and who decides

The plugin is an ordinary MCP client of the Corvus it sits in. It signs in to
`/mcp` with the token of `GET /api/mcp` (this computer only), so it gets exactly
the tools the operator's switches in Settings, Connections, MCP allow, and the
same question before anything that changes the aircraft. It never talks to the
vehicle by another road. The MCP server is off by default; the chat says so and
has a button that switches it on. What changes the aircraft stays off there
until you allow it.

A model that keeps asking for tools is ended after 12 rounds in one turn.

## What it is

A Rust crate (`corvus_plugin_ai_chat`) built on `corvus_api`, the plugin API of
Corvus GCS. It is not part of the program: it is published as a shared
library from this repository and installed from Settings, Plugins. There is no JavaScript.

- `src/model.rs`: the rules, free of the interface and the network: the
  address, both dialects' requests and answers, the tools as the model is
  offered them, how much of a long chat is sent again.
- `src/client.rs`: one HTTP or HTTPS request that Stop ends at once.
- `src/agent.rs`: one turn on a thread of its own (ask, run the tools, ask
  again) and the MCP client.
- `src/lib.rs`: the plugin, its options and the chat.
- `src/view.rs`: what is drawn.
- `lang/`: German, French and the dialect file.

## What it needs

A Corvus GCS with `/mcp` and `corvus_api::plugin::options::OptionSpec::secret`.

## Change it

Corvus GCS carries this repository as a submodule at `plugins/ai-chat`, and the
crate is a member of its workspace:

```bash
git clone --recurse-submodules https://github.com/M-BSquared/CorvusGCS.git
cd CorvusGCS
./run.sh
```

Settings, Plugins then offers the copy built there. A dependency must be one
that `Cargo.lock` of Corvus GCS already holds.

## Publish it

Commit and push to `main`. The workflow tests the plugin, builds it for the
newest published Corvus GCS on Linux, macOS and Windows, lets that program load
it, and attaches it to the release `corvus-<version>` of this repository, where
Corvus GCS of that version finds it. A new version of Corvus GCS gets its build
within hours.

## Tests

From the Corvus checkout:

```bash
cargo test -p corvus_plugin_ai_chat
```

## Licence

Sustainable Use License, see [LICENSE.md](LICENSE.md). Copyright (c) 2026
Maximilian Böck.
