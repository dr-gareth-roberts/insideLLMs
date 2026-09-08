# insideLLMs Tools

Use **Run insideLLMs probes** in Python CodeLens actions or **insideLLMs: Run
Probes** in the command palette. Install the Python CLI separately and make
`insidellms` available on the editor's process PATH.

The invoking file selects its workspace folder, including in multi-root
workspaces. Palette invocation uses the active file or asks you to select a
folder. Workspace-scoped `insidellms.harnessConfigPath` (default
`ci/harness.yaml`) and `insidellms.runDir` (default `.tmp/runs/ide`) resolve
against that folder. Files outside the workspace do not select another folder.

The command requires a trusted workspace. It creates a VS Code process task
with separate executable and arguments; spaces, quotes and shell metacharacters
in paths remain literal. The task runs `insidellms harness CONFIG --run-dir DIR
--overwrite --skip-report`, so it replaces artifacts in the configured run
directory. The task terminal shows CLI output and exit status.

## Build and install locally

Use Node **22.23.1** and npm **10.9.8**, matching `.nvmrc`, manifest engines and
lockfile. From this directory:

```sh
npm ci --ignore-scripts
npm run build
npm test
npm run package
npm run verify-package
code --install-extension ./insidellms-tools-0.1.0.vsix
```

The build has no runtime npm dependencies. TypeScript and exact VS Code/Node
types compile the extension. Microsoft's pinned `@vscode/vsce` generates the
supported VSIX manifest/container. Pinned `yauzl` and `yazl` read and normalize
that container: sorted entries, fixed timestamps/modes, no compression, no
source paths or source maps. They are packaging-only dependencies. Installation
scripts are disabled during clean installs; no native helper is needed to
package. The lockfile records the complete transitive graph.

The archive has exactly six files: VSIX metadata plus package.json,
dist/extension.js, readme.md and LICENSE.txt. Archive validation rejects sources,
dependencies, credentials and unexpected entries. `verify-package` performs two
clean offline `npm ci` builds with differing source mtimes and timezones, runs
tests in both copies, then compares SHA-256 of the whole VSIX bytes. Its offline
installs require an npm cache primed by the initial clean install. Temporary
verification builds are removed afterward.

The CI workflow has read-only repository permissions and runs these local
checks. It does not publish to the Marketplace or upload the VSIX.

## Real editor-host check

Unit tests mock the VS Code boundary; they prove launch argument/workspace/trust
contracts, not editor activation or actual Python execution. For the separate
real-host integration gate, build the package, install the Python project's
offline development environment at `.venv`, and set a local editor executable:

```sh
INSIDELLMS_EDITOR_EXECUTABLE='/Applications/Visual Studio Code.app/Contents/MacOS/Code' npm run test:editor
```

This starts a separate editor with isolated user data and extensions, loads the
unpacked local VSIX, explicitly trusts only its temporary test workspace, and
runs one DummyModel harness example. It checks activation, process exit status,
completed manifest and records. Logs and JSON proof are retained under
`.editor-test/`; the normal editor profile is not changed. No provider API call
or publication is made. Missing/unlaunchable editor hosts fail this gate and
must be reported as unavailable proof, not replaced with unit-test results.

See [VS Code process execution API](https://code.visualstudio.com/api/references/vscode-api#ProcessExecution)
and [official packaging documentation](https://code.visualstudio.com/api/working-with-extensions/publishing-extension).
