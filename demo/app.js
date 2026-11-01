(() => {
  'use strict';
  const data = window.BBCSPORT_DEMO;
  const $ = id => document.getElementById(id);
  if (!data || data.schema_version !== 1) {
    $('fatal-error').hidden = false;
    $('fatal-error').textContent = 'The saved experiment recording is missing. Run the experiment with --export-demo, then reopen the demo.';
    return;
  }
  const categoryName = index => data.categories[index][0].toUpperCase() + data.categories[index].slice(1);
  const methods = ['supervised', 'cotraining', 'colabel_flat', 'colabel_grouped'];
  const methodLabels = ['Known answers only', 'Older approach: co-training', 'Co-labeling: one layer', 'Co-labeling: two layers'];
  const trained = data.methods.colabel_grouped;
  const lastRound = trained.states.length - 1;
  const trainArticles = data.articles.filter(a => a.role === 'train');
  const testArticles = data.articles.filter(a => a.role === 'test');
  const knownArticles = trainArticles.filter(a => a.known);
  const unknownArticles = trainArticles.filter(a => !a.known);
  let chapter = 'start', round = 0, selected = testArticles[0], focus = 2, receiver = 0;
  let timer = null, revealed = false, requestNumber = 0;
  const titles = new Map();

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function color(node, category) { node.style.setProperty('--sport-color', `var(--sport${category})`); return node; }
  function argmax(values) { return values.indexOf(Math.max(...values)); }
  function prediction(article, atRound) {
    return trained.states[atRound][`${article.role}_predictions`][article.row];
  }
  function pause() {
    if (timer) clearInterval(timer);
    timer = null;
    $('round-play').textContent = 'Play';
    $('round-play').setAttribute('aria-pressed', 'false');
  }
  function go(nextChapter) {
    chapter = nextChapter;
    if (chapter !== 'learn') pause();
    document.querySelectorAll('[data-page]').forEach(page => { page.hidden = page.dataset.page !== chapter; });
    document.querySelectorAll('[data-chapter]').forEach(button => {
      const active = button.dataset.chapter === chapter;
      button.classList.toggle('active', active);
      if (active) button.setAttribute('aria-current', 'step'); else button.removeAttribute('aria-current');
    });
    if (chapter === 'article') renderArticle();
    history.replaceState(null, '', `#${chapter}`);
    window.scrollTo({ top: 0, behavior: 'instant' });
  }
  document.querySelectorAll('[data-chapter]').forEach(button => button.addEventListener('click', () => go(button.dataset.chapter)));
  document.querySelectorAll('[data-go]').forEach(button => button.addEventListener('click', () => go(button.dataset.go)));

  async function articleText(article) {
    if (titles.has(article.id)) return titles.get(article.id);
    try {
      const response = await fetch(`/api/article?id=${article.id}`);
      if (!response.ok) return { available: false };
      const result = await response.json();
      titles.set(article.id, result);
      return result;
    } catch { return { available: false }; }
  }

  function renderKnown() {
    $('known-examples').replaceChildren();
    knownArticles.forEach(article => {
      const row = el('div', 'known-example');
      const mark = color(el('span', 'sport-mark'), article.answer);
      mark.style.background = `var(--sport${article.answer})`;
      const text = el('div', 'example-text');
      const title = el('strong', '', `Article ${article.id + 1}`);
      text.append(title, el('small', '', categoryName(article.answer)));
      row.append(mark, text);
      $('known-examples').append(row);
      articleText(article).then(result => { if (result.available) { title.textContent = result.title; title.title = result.title; } });
    });
  }

  function renderSheets() {
    const target = $('candidate-sheets');
    target.replaceChildren();
    const detail = trained.states[round].details;
    const entry = detail.categories?.find(item => item.category === focus);
    if (!entry) {
      $('sheet-explanation').textContent = 'At round 0, the readers have learned from the ten human answers only.';
      target.append(el('p', 'empty-sheets', 'Advance one round to see the real alternative answer sheets and their learned influence.'));
      return;
    }
    const view = entry.views[receiver];
    const sorted = [...view.sheets].sort((a, b) => b.weight - a.weight).slice(0, 8);
    const preview = detail.preview_training_rows.map(row => trainArticles[row]);
    const sport = categoryName(focus), initial = sport[0];
    $('sheet-explanation').textContent = `Reader ${receiver === 0 ? 'A' : 'B'} has ${view.candidate_count} alternatives for “${sport} or another sport”. Showing the eight with greatest influence.`;
    const wrapper = el('div', 'sheet-scroll');
    const table = el('table', 'sheets-table');
    table.setAttribute('aria-label', `Actual ${sport} candidate labelings and shares of learned coefficient weight`);
    const head = el('thead'), header = el('tr');
    ['Answer sheet', 'Influence', ...preview.map(a => `#${a.id + 1}`)].forEach(text => {
      const th = el('th', '', text); th.scope = 'col'; header.append(th);
    });
    head.append(header); table.append(head);
    const body = el('tbody');
    sorted.forEach(sheet => {
      const row = el('tr');
      row.append(el('td', '', `Round ${sheet.origin_round} · ${Math.round(sheet.fraction * 100)}% rule`));
      row.append(el('td', 'number', `${(sheet.share * 100).toFixed(1)}%`));
      sheet.answers.forEach(answer => {
        const cell = color(el('td', answer === 1 ? 'sheet-yes' : '', answer === 1 ? initial : '–'), focus);
        cell.setAttribute('aria-label', answer === 1 ? sport : 'Another sport');
        row.append(cell);
      });
      body.append(row);
    });
    table.append(body); wrapper.append(table); target.append(wrapper);
    target.append(el('p', 'sheet-footnote', `${initial} = ${sport}; – = another sport. Influence is a share of learned coefficient weight, not confidence. The “% rule” sets how many unknown articles a sheet labels as ${sport.toLowerCase()}.`));
  }

  function renderRound() {
    const state = trained.states[round], previous = trained.states[Math.max(0, round - 1)];
    $('round-number').textContent = round;
    $('round-context').textContent = `of ${lastRound}`;
    $('round-back').disabled = round === 0;
    $('round-next').disabled = round === lastRound;
    const changes = unknownArticles.filter(a => state.train_predictions[a.row] !== previous.train_predictions[a.row]).length;
    $('changed-readout').textContent = round ? `${changes} of the 360 guesses changed since the previous round` : 'Starting guesses: learned from ten supplied answers';
    $('round-score').textContent = `Test ranking score: ${state.metrics.map.toFixed(2)} / 100`;
    for (let view = 0; view < 2; view++) {
      const count = unknownArticles.filter(a => argmax(state.train_views[view][a.row]) !== argmax(previous.train_views[view][a.row])).length;
      $(view ? 'b-changes' : 'a-changes').textContent = round ? `${count} guesses changed this round` : 'Starts with the same ten answers';
    }
    $('article-grid').replaceChildren();
    Array.from({ length: 60 }, (_, index) => unknownArticles[Math.floor(index * (unknownArticles.length - 1) / 59)]).forEach(article => {
      const guessed = state.train_predictions[article.row];
      const button = color(el('button', 'article-cell', `#${article.id + 1}`), guessed);
      button.type = 'button';
      button.setAttribute('aria-label', `Article ${article.id + 1}, currently guessed ${categoryName(guessed)}. Inspect article.`);
      button.title = `Article ${article.id + 1}: guessed ${categoryName(guessed)}`;
      button.addEventListener('click', () => { selected = article; revealed = false; $('article-select').value = article.id; go('article'); });
      $('article-grid').append(button);
    });
    renderSheets();
  }
  $('round-back').addEventListener('click', () => { pause(); round = Math.max(0, round - 1); renderRound(); });
  $('round-next').addEventListener('click', () => { pause(); round = Math.min(lastRound, round + 1); renderRound(); });
  $('round-reset').addEventListener('click', () => { pause(); round = 0; renderRound(); });
  $('round-play').addEventListener('click', () => {
    if (timer) { pause(); return; }
    if (round === lastRound) round = 0;
    $('round-play').textContent = 'Pause';
    $('round-play').setAttribute('aria-pressed', 'true');
    renderRound();
    timer = setInterval(() => { round++; renderRound(); if (round >= lastRound) pause(); }, 1500);
  });
  $('focus-category').replaceChildren(...data.categories.map((_, index) => {
    const option = el('option', '', categoryName(index)); option.value = index; return option;
  }));
  $('focus-category').value = focus;
  $('focus-category').addEventListener('change', event => { focus = Number(event.target.value); renderSheets(); });
  $('focus-view').addEventListener('change', event => { receiver = Number(event.target.value); renderSheets(); });
  $('sport-legend').replaceChildren(...data.categories.map((_, index) => {
    const item = el('span'); item.append(color(el('i'), index), document.createTextNode(categoryName(index))); return item;
  }));

  const articleOptions = new Map();
  for (const [label, articles] of [['New test articles (never used for training)', testArticles], ['Unknown training articles (available during learning)', unknownArticles]]) {
    const group = document.createElement('optgroup'); group.label = label;
    articles.forEach(article => {
      const option = el('option', '', `Article #${article.id + 1} · ${article.words.flat().slice(0, 3).join(', ')}`);
      option.value = article.id; group.append(option); articleOptions.set(article.id, option);
    });
    $('article-select').append(group);
  }
  $('article-select').value = selected.id;
  $('article-select').addEventListener('change', event => { selected = data.articles.find(a => a.id === Number(event.target.value)); revealed = false; renderArticle(); });
  $('changed-example').addEventListener('click', () => {
    const changed = testArticles.filter(a => prediction(a, 0) !== prediction(a, lastRound));
    if (!changed.length) return;
    const current = changed.findIndex(a => a.id === selected.id);
    selected = changed[(current + 1) % changed.length];
    $('article-select').value = selected.id;
    revealed = false; renderArticle();
  });
  $('changed-example').disabled = !testArticles.some(a => prediction(a, 0) !== prediction(a, lastRound));
  $('reveal-answer').addEventListener('click', () => { revealed = !revealed; renderAnswer(); });

  function renderAnswer() {
    $('reveal-answer').textContent = revealed ? 'Hide the answer' : 'Reveal the real answer';
    if (!revealed) { $('real-answer').textContent = ''; return; }
    const before = prediction(selected, 0) === selected.answer;
    const after = prediction(selected, lastRound) === selected.answer;
    $('real-answer').textContent = `${categoryName(selected.answer)} · Starting guess ${before ? 'correct' : 'incorrect'}; learned guess ${after ? 'correct' : 'incorrect'}.`;
  }
  function renderArticle() {
    const request = ++requestNumber;
    $('article-page-eyebrow').textContent = selected.role === 'test' ? '03 / AN ARTICLE THE READERS HAVE NOT PRACTICED ON' : '03 / INSPECT AN UNLABELED TRAINING ARTICLE';
    $('article-page-lead').textContent = selected.role === 'test' ? 'Choose a new test article. Compare the starting guess with the learned prediction, then reveal its answer.' : 'This training article was available without its answer. Compare its starting guess with its learned prediction.';
    $('article-role').textContent = selected.role === 'test' ? 'NEW TEST ARTICLE · NEVER USED FOR TRAINING' : 'UNKNOWN TRAINING ARTICLE · USED WITHOUT ITS ANSWER';
    $('article-title').textContent = `Article #${selected.id + 1}`;
    $('article-excerpt').textContent = 'The original article text is optional; its recorded word clues and predictions are available offline.';
    articleText(selected).then(result => {
      if (request !== requestNumber) return;
      if (result.available) { $('article-title').textContent = result.title; $('article-excerpt').textContent = result.excerpt; }
    });
    ['words-a', 'words-b'].forEach((id, view) => $(id).replaceChildren(...selected.words[view].map(word => el('span', '', word))));
    $('prediction-before').textContent = categoryName(prediction(selected, 0));
    $('prediction-after').textContent = categoryName(prediction(selected, lastRound));
    const scores = trained.states[lastRound][`${selected.role}_views`];
    $('individual-predictions').textContent = `A: ${categoryName(argmax(scores[0][selected.row]))} · B: ${categoryName(argmax(scores[1][selected.row]))} · combined result above`;
    renderAnswer();
  }

  function bars(container, values, deviations = null) {
    container.replaceChildren();
    methods.forEach((method, index) => {
      const row = el('div', `result-row${method === 'colabel_grouped' ? ' featured' : ''}`);
      const label = el('div', 'result-label');
      const variability = deviations?.[method];
      label.append(el('span', '', methodLabels[index]), el('span', 'result-value', `${values[method].toFixed(2)}${variability == null ? '' : ` ± ${variability.toFixed(2)}`}`));
      const track = el('div', 'result-track'), fill = el('div', 'result-fill');
      fill.style.width = `${Math.max(0, Math.min(100, values[method]))}%`;
      track.append(fill); row.append(label, track); container.append(row);
    });
  }
  const report = data.experiment;
  bars($('our-results'), Object.fromEntries(methods.map(method => [method, report.methods[method].map_mean])), Object.fromEntries(methods.map(method => [method, report.methods[method].map_std])));
  bars($('paper-results'), report.paper.map, report.paper.std);
  $('our-results').setAttribute('aria-label', methods.map((m, index) => `${methodLabels[index]}: ${report.methods[m].map_mean.toFixed(2)} out of 100`).join('; '));
  $('results-runs').textContent = report.config.runs === 1 ? 'One predeclared split · seed ' + report.seeds[0] : `Mean ± variation across ${report.config.runs} predeclared splits`;
  $('accuracy-results').replaceChildren(...methods.map((method, index) => {
    const line = el('div', 'accuracy-line'); line.append(el('span', '', methodLabels[index]), el('span', 'number', `${report.methods[method].accuracy_mean.toFixed(2)}%`)); return line;
  }));
  const facts = [
    ['Human answers', `${report.config.labels_per_class} per sport · ${report.config.labels_per_class * 5} total`],
    ['Data split', '370 training articles · 367 separate test articles'],
    ['Repeated runs', String(report.config.runs)], ['Seeds', report.seeds.join(', ')],
    ['Replay seed', String(data.recording_seed)], ['Learning rounds', `Up to ${report.config.rounds}; stops earlier when stable`],
    ['Word features', 'Original term counts · random vocabulary split · separate normalization'],
    ['Fixed C', String(report.config.c)], ['Generated', new Date(report.created_utc).toLocaleString()],
  ];
  facts.forEach(([label, value]) => $('experiment-facts').append(el('dt', '', label), el('dd', '', value)));
  const fits = report.runs.flatMap(run => methods.map(method => run.methods[method].solver));
  const fitCount = fits.reduce((sum, item) => sum + item.fits, 0), converged = fits.reduce((sum, item) => sum + item.converged, 0);
  $('solver-note').textContent = fitCount === converged ? 'All recorded optimization fits reached the requested numerical tolerance.' : `${converged} of ${fitCount} recorded optimization fits reached the requested numerical tolerance; the others reached the iteration limit. See REPRODUCTION.md for the numerical settings and limits.`;
  $('run-caption').textContent = `${report.config.runs} measured ${report.config.runs === 1 ? 'run' : 'runs'} · replay seed ${data.recording_seed}`;
  $('notes-button').addEventListener('click', () => { const notes = $('presenter-notes'); notes.hidden = !notes.hidden; $('notes-button').setAttribute('aria-expanded', String(!notes.hidden)); });
  $('fullscreen').addEventListener('click', async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch { /* Browser may disallow fullscreen; presentation remains usable. */ }
  });
  document.addEventListener('fullscreenchange', () => { $('fullscreen').textContent = document.fullscreenElement ? 'Exit full screen' : 'Full screen'; });
  document.addEventListener('keydown', event => {
    if (/INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
    if (/^[1-4]$/.test(event.key)) { go(['start', 'learn', 'article', 'results'][Number(event.key) - 1]); event.preventDefault(); }
    if (chapter !== 'learn') return;
    if (event.key === 'ArrowRight') { $('round-next').click(); event.preventDefault(); }
    if (event.key === 'ArrowLeft') { $('round-back').click(); event.preventDefault(); }
    if (event.code === 'Space') { $('round-play').click(); event.preventDefault(); }
  });
  renderKnown(); renderRound();
  go(['start', 'learn', 'article', 'results'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'start');
})();
