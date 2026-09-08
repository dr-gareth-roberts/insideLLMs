const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');
const { entries, validate } = require('./archive.cjs');

async function main() {
  const root = path.resolve(__dirname, '..');
  const editor = process.env.INSIDELLMS_EDITOR_EXECUTABLE;
  if (!editor || !fs.existsSync(editor)) {
    throw new Error('Set INSIDELLMS_EDITOR_EXECUTABLE to a local VS Code executable; real host proof unavailable');
  }
  const manifest = require('../package.json');
  const archivePath = path.join(root, `${manifest.name}-${manifest.version}.vsix`);
  const archiveSha256 = crypto.createHash('sha256').update(fs.readFileSync(archivePath)).digest('hex');
  const content = await entries(archivePath);
  validate(content);
  const output = path.join(root, '.editor-test');
  fs.mkdirSync(output, { recursive: true });
  const session = fs.mkdtempSync(path.join(output, 'session-'));
  // macOS Unix-domain sockets have a short path limit; keep user data short.
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'illm-'));
  fs.writeFileSync(path.join(session, 'profile-location.txt'), profile);
  const extension = path.join(session, 'extension');
  for (const [name, bytes] of content) {
    if (!name.startsWith('extension/')) continue;
    const target = path.join(session, name);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, bytes);
  }
  const workspace = path.join(session, 'workspace $(literal); spaces');
  fs.mkdirSync(path.join(workspace, 'ci'), { recursive: true });
  fs.writeFileSync(path.join(workspace, 'ci/harness.yaml'), JSON.stringify({
    models: [{ type: 'dummy' }], probes: [{ type: 'logic' }],
    dataset: { format: 'jsonl', path: 'dataset.jsonl' }
  }));
  fs.writeFileSync(path.join(workspace, 'ci/dataset.jsonl'), '"If all cats are animals, is a cat an animal?"\n');
  fs.writeFileSync(path.join(workspace, 'prompt.py'), 'prompt = "offline test"\n');
  const repository = path.resolve(root, '../..');
  const testEntry = path.join(session, 'host-tests.cjs');
  fs.writeFileSync(testEntry, `exports.run = async () => {
    try { await require(${JSON.stringify(path.join(root, 'tests/editor-host.cjs'))}).run(); }
    catch (error) {
      require('node:fs').writeFileSync(${JSON.stringify(path.join(session, 'failure.txt'))}, String(error.stack || error));
      throw error;
    }
  };`);
  const log = fs.openSync(path.join(session, 'host.log'), 'w');
  const child = spawn(editor, [
    '--user-data-dir', profile,
    '--extensions-dir', path.join(session, 'installed-extensions'),
    '--extensionDevelopmentPath', extension,
    '--extensionTestsPath', testEntry,
    // This isolated test supplies PATH explicitly; do not execute user login scripts.
    '--force-disable-user-env',
    '--disable-extensions', '--disable-workspace-trust', '--skip-welcome',
    '--skip-release-notes', '--disable-updates', workspace
  ], { env: {
    ...process.env, PATH: `${path.join(repository, '.venv/bin')}${path.delimiter}${process.env.PATH}`,
    MPLCONFIGDIR: path.join(repository, '.tmp/test-cache/matplotlib'),
    XDG_CACHE_HOME: path.join(repository, '.tmp/test-cache/xdg'),
    INSIDELLMS_EDITOR_TEST_OUTPUT: session
  }, stdio: ['ignore', log, log] });
  fs.closeSync(log);
  const timer = setTimeout(() => child.kill('SIGTERM'), 90000);
  const code = await new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', resolve);
  });
  clearTimeout(timer);
  const proof = path.join(session, 'proof.json');
  if (code !== 0 || !fs.existsSync(proof)) {
    throw new Error(`Editor host failed (exit ${code}); retained logs: ${session}`);
  }
  const verified = { ...JSON.parse(fs.readFileSync(proof, 'utf8')), archiveSha256 };
  fs.writeFileSync(proof, JSON.stringify(verified, null, 2));
  console.log(JSON.stringify(verified, null, 2));
  console.log(`Editor proof retained: ${session}`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
