const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function load(trusted = true) {
  const state = { tasks: [], errors: [], scopes: [] };
  const folders = ['/first', '/second $(`literal`)'].map(fsPath => ({ uri: { fsPath } }));
  const vscode = {
    workspace: {
      isTrusted: trusted,
      workspaceFolders: folders,
      getWorkspaceFolder: uri => folders.find(folder => uri.fsPath.startsWith(folder.uri.fsPath)),
      getConfiguration: (_, uri) => {
        state.scopes.push(uri);
        return { get: (key, fallback) => key === 'harnessConfigPath' ? 'config $(literal); x.yaml' : fallback };
      }
    },
    window: {
      showErrorMessage: message => state.errors.push(message),
      showWorkspaceFolderPick: async () => folders[1]
    },
    languages: { registerCodeLensProvider: () => ({ dispose() {} }) },
    commands: { registerCommand: (_, callback) => { state.command = callback; return { dispose() {} }; } },
    Uri: { parse: value => ({ fsPath: value.replace('file://', '') }) },
    ProcessExecution: class { constructor(process, args, options) { Object.assign(this, { process, args, options }); } },
    Task: class { constructor(definition, scope, name, source, execution) { Object.assign(this, { definition, scope, name, source, execution }); } },
    tasks: { executeTask: async task => { state.tasks.push(task); return task; } }
  };
  const module = { exports: {} };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../dist/extension.js'), 'utf8'), {
    require: name => name === 'vscode' ? vscode : require(name), module, exports: module.exports
  });
  module.exports.activate({ subscriptions: [] });
  return { state, folders };
}

test('invoking file chooses its workspace and passes metacharacters literally', async () => {
  const { state, folders } = load();
  await state.command(`file://${folders[1].uri.fsPath}/prompt.py`);
  const task = state.tasks[0];
  assert.equal(task.scope, folders[1]);
  assert.equal(task.execution.process, 'insidellms');
  assert.equal(task.execution.options.cwd, folders[1].uri.fsPath);
  assert.equal(task.execution.args[1], `${folders[1].uri.fsPath}/config $(literal); x.yaml`);
  assert.equal(state.scopes[0], folders[1].uri);
});

test('untrusted workspaces never dispatch a process', async () => {
  const { state } = load(false);
  await state.command('file:///first/prompt.py');
  assert.equal(state.tasks.length, 0);
  assert.equal(state.scopes.length, 0);
  assert.equal(state.errors.length, 1);
});

test('an outside file does not fall back to another workspace', async () => {
  const { state } = load();
  await state.command('file:///outside/prompt.py');
  assert.equal(state.tasks.length, 0);
});

test('palette invocation selects a workspace when multiple folders are open', async () => {
  const { state, folders } = load();
  await state.command();
  assert.equal(state.tasks[0].scope, folders[1]);
});

test('package exposes compiled entrypoint and a runtime allowlist', () => {
  const manifest = require('../package.json');
  assert.ok(fs.existsSync(path.join(__dirname, '..', manifest.main)));
  assert.deepEqual(manifest.files, ['dist/extension.js', 'README.md', 'LICENSE']);
  assert.equal(manifest.capabilities.untrustedWorkspaces.supported, false);
  const properties = manifest.contributes.configuration.properties;
  assert.equal(properties['insidellms.harnessConfigPath'].scope, 'resource');
  assert.equal(properties['insidellms.runDir'].scope, 'resource');
});
