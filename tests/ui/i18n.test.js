const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const htmlPath = path.join(__dirname, '../../eujeno/ui/static/index.html');
const html = fs.readFileSync(htmlPath, 'utf8');

// Extract TRANSLATIONS object between markers
const begin = html.indexOf('// BEGIN_TRANSLATIONS');
const end = html.indexOf('// END_TRANSLATIONS');
assert.ok(begin !== -1, 'BEGIN_TRANSLATIONS marker not found');
assert.ok(end !== -1, 'END_TRANSLATIONS marker not found');

const block = html.slice(begin, end);
// Extract just the object literal: from 'const TRANSLATIONS = ' to end
const objStart = block.indexOf('{');
const objText = block.slice(objStart);
// We need to find the balanced closing brace
let depth = 0, objEnd = 0;
for (let i = 0; i < objText.length; i++) {
  if (objText[i] === '{') depth++;
  else if (objText[i] === '}') { depth--; if (depth === 0) { objEnd = i + 1; break; } }
}
const objLiteral = objText.slice(0, objEnd);
const TRANSLATIONS = new Function('return (' + objLiteral + ')')();

test('languages are exactly it,en,fr,de,es', () => {
  const langs = Object.keys(TRANSLATIONS).sort();
  assert.deepEqual(langs, ['de', 'en', 'es', 'fr', 'it']);
});

test('all languages have identical key sets', () => {
  const allKeys = new Set();
  Object.values(TRANSLATIONS).forEach(d => Object.keys(d).forEach(k => allKeys.add(k)));
  const union = [...allKeys].sort();
  console.log('Total keys:', union.length);
  for (const lang of Object.keys(TRANSLATIONS)) {
    const langKeys = Object.keys(TRANSLATIONS[lang]).sort();
    for (const key of union) {
      assert.ok(langKeys.includes(key), `Language "${lang}" is missing key "${key}"`);
    }
  }
});

test('no value is an empty string', () => {
  for (const [lang, dict] of Object.entries(TRANSLATIONS)) {
    for (const [key, val] of Object.entries(dict)) {
      assert.ok(typeof val === 'string' && val.length > 0, `Language "${lang}", key "${key}" has empty value`);
    }
  }
});
