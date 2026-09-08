const assert = require('node:assert/strict');
const { test } = require('node:test');
const { validate, allowed } = require('../scripts/archive.cjs');

test('archive allowlist rejects source, dependencies, credentials and missing entrypoint', () => {
  for (const extra of ['extension/src/extension.ts', 'extension/node_modules/x.js', 'extension/.env']) {
    const content = new Map(allowed.map(name => [name, Buffer.from('{}')]));
    content.set(extra, Buffer.from('unwanted'));
    assert.throws(() => validate(content), /Unexpected VSIX/);
  }
  const missing = new Map(allowed.filter(name => !name.endsWith('extension.js')).map(name => [name, Buffer.from('{}')]));
  assert.throws(() => validate(missing), /Unexpected VSIX/);
});
