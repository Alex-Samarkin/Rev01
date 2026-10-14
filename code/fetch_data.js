#!/usr/bin/env node
/**
 * fetch_data.js — загрузка открытых данных о заболеваемости по городам/регионам.
 *
 * Запуск:
 *     node code/fetch_data.js            # скачать всё в data/raw
 *     node code/fetch_data.js --force    # перекачать поверх существующих файлов
 *
 * Скрипт не требует зависимостей (только стандартная библиотека Node.js) и
 * специально написан так, чтобы обходить ограничения окружений, в которых
 * TLS-стек PowerShell недоступен: используется TLS-реализация Node.js.
 *
 * Для каждого файла печатается размер и SHA-256 — эти значения попадают в
 * статью как подтверждение того, что использовалась именно эта версия данных.
 */
'use strict';

const https = require('https');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const ROOT = path.resolve(__dirname, '..');
const RAW = path.join(ROOT, 'data', 'raw');
const FORCE = process.argv.includes('--force');

/** Манифест источников. */
const SOURCES = [
  {
    file: 'nyc_cases_by_day.csv',
    url: 'https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/cases-by-day.csv',
    source: 'New York City Department of Health and Mental Hygiene (NYC DOHMH), ' +
            'репозиторий nychealth/coronavirus-data, файл trends/cases-by-day.csv',
    territory: 'Нью-Йорк (США)',
  },
  {
    file: 'pcm_dpc_province.csv',
    url: 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv',
    source: 'Dipartimento della Protezione Civile (Италия), репозиторий pcm-dpc/COVID-19, ' +
            'файл dati-province/dpc-covid19-ita-province.csv',
    territory: 'провинции Италии (в т.ч. Милан)',
  },
  {
    file: 'ru_regions_infected.csv',
    url: 'https://raw.githubusercontent.com/artem-kuchumov/covid-russia/master/data-infected.csv',
    source: 'Набор данных «COVID-19 в России» (artem-kuchumov/covid-russia), ' +
            'data-infected.csv — накопленное число случаев по субъектам РФ',
    territory: 'субъекты РФ (в т.ч. Москва, Псковская, Калининградская области)',
  },
  {
    file: 'in_delhi_cumulative.csv',
    url: 'https://raw.githubusercontent.com/IBM/covid19-india-data/main/data/csv/DL/DL_cumulative.csv',
    source: 'IBM covid19-india-data, data/csv/DL/DL_cumulative.csv — накопленные случаи, Дели',
    territory: 'Дели (Индия)',
  },
];

function sha256(buf) {
  return crypto.createHash('sha256').update(buf).digest('hex');
}

function download(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error('too many redirects'));
    https
      .get(url, { headers: { 'User-Agent': 'covid-wave-research/1.0' } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          return resolve(download(res.headers.location, redirects + 1));
        }
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error('HTTP ' + res.statusCode + ' for ' + url));
        }
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => resolve(Buffer.concat(chunks)));
      })
      .on('error', reject);
  });
}

(async () => {
  fs.mkdirSync(RAW, { recursive: true });
  const manifest = [];
  for (const s of SOURCES) {
    const dest = path.join(RAW, s.file);
    let buf;
    if (fs.existsSync(dest) && !FORCE) {
      buf = fs.readFileSync(dest);
      console.log(`[skip] ${s.file} (уже есть, ${buf.length} байт)`);
    } else {
      process.stdout.write(`[get ] ${s.file} ... `);
      buf = await download(s.url);
      fs.writeFileSync(dest, buf);
      console.log(`${buf.length} байт`);
    }
    manifest.push({
      file: s.file,
      url: s.url,
      source: s.source,
      territory: s.territory,
      bytes: buf.length,
      sha256: sha256(buf),
    });
  }
  const out = path.join(ROOT, 'data', 'manifest.json');
  fs.writeFileSync(out, JSON.stringify({ downloaded_at_utc: new Date().toISOString(), files: manifest }, null, 2));
  console.log('\nМанифест записан: ' + out);
  for (const m of manifest) console.log(`  ${m.file.padEnd(28)} ${String(m.bytes).padStart(9)}  ${m.sha256.slice(0, 16)}…`);
})().catch((e) => {
  console.error('ОШИБКА: ' + e.message);
  process.exit(1);
});
