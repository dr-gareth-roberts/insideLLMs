const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');

exports.run = async function run() {
  const extension = vscode.extensions.all.find(item => item.id.toLowerCase() === 'insidellms.insidellms-tools');
  assert.ok(extension, 'Packaged extension was discovered by real editor host');
  await extension.activate();
  assert.ok(extension.isActive);
  const folder = vscode.workspace.workspaceFolders[0];
  const completed = new Promise((resolve, reject) => {
    const timer = setTimeout(() => { listener.dispose(); reject(new Error('Harness task timeout')); }, 60000);
    const listener = vscode.tasks.onDidEndTaskProcess(event => {
      if (event.execution.task.source !== 'insideLLMs') return;
      clearTimeout(timer);
      listener.dispose();
      resolve(event.exitCode);
    });
  });
  await vscode.commands.executeCommand('insidellms.runProbes', vscode.Uri.joinPath(folder.uri, 'prompt.py'));
  assert.equal(await completed, 0);
  const runDir = path.join(folder.uri.fsPath, '.tmp/runs/ide');
  const manifest = JSON.parse(fs.readFileSync(path.join(runDir, 'manifest.json'), 'utf8'));
  const records = fs.readFileSync(path.join(runDir, 'records.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(manifest.run_completed, true);
  assert.equal(records.length, 1);
  fs.writeFileSync(path.join(process.env.INSIDELLMS_EDITOR_TEST_OUTPUT, 'proof.json'), JSON.stringify({
    editor: vscode.version, extension: extension.id, activated: extension.isActive,
    taskExitCode: 0, runCompleted: true, records: records.length, runDir
  }, null, 2));
};
