(() => {
  'use strict';
  const data = window.BBCSPORT_STORY;
  const $ = id => document.getElementById(id);
  const video = $('story-video');
  function error(message) { $('player-error').hidden = false; $('player-error').textContent = message; }
  if (!data || data.schema_version !== 1) {
    error('Replay data is missing. Run python render_presentation.py, then reopen the demo.');
    return;
  }
  const descriptions = [
    'Both classifiers have already been trained on the ten known labels. The colored grid shows their starting predictions for the 360 unlabeled articles.',
    `${data.rounds[1].changed_count} predictions change after round 1. The article positions remain fixed so that the changes can be followed.`,
    'Reader B creates eleven new football label lists. Reader A retains eleven earlier lists and combines the twenty-two alternatives in groups.',
    'The unchanged optimizer jointly fits the classifier and learns the sheets’ influence from Reader A’s word patterns. Bars show sampled actual updates. Influence is a share of learned weight, not confidence.',
    `${data.rounds[2].changed_count} predictions change in round 2. The same process is repeated for the other four sports.`,
    'Rounds 3–8 are shown faster. The saved recording has no predicted-sport changes in its final two rounds.',
    'Training is finished. Return to the results slide, or open the measured comparison below.',
  ];
  const shortNames = ['Starting predictions', 'Round 1', 'Create and exchange', 'Learn influence', 'Update predictions', 'Later rounds', 'Finish'];
  const buttons = data.chapters.map((chapter, i) => {
    const button = document.createElement('button');
    button.type = 'button'; button.textContent = `${i + 1}. ${shortNames[i]}`;
    button.addEventListener('click', () => seek(chapter.start));
    $('chapters').append(button);
    return button;
  });
  const format = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
  function sceneIndex() {
    return Math.max(0, data.chapters.findLastIndex(chapter => video.currentTime + .01 >= chapter.start));
  }
  function seek(time) {
    video.pause(); video.currentTime = Math.max(0, Math.min(data.duration - .05, time)); update();
  }
  function update() {
    const i = sceneIndex();
    $('play').textContent = video.paused ? 'Play' : 'Pause';
    $('play').setAttribute('aria-pressed', String(!video.paused));
    $('time').textContent = `${format(video.currentTime)} / ${format(data.duration)}`;
    $('scene-title').textContent = data.chapters[i].title;
    $('scene-detail').textContent = descriptions[i];
    $('previous').disabled = i === 0 && video.currentTime < .1;
    $('next').disabled = i === data.chapters.length - 1;
    buttons.forEach((button, j) => { if (i === j) button.setAttribute('aria-current', 'step'); else button.removeAttribute('aria-current'); });
  }
  function toggle() {
    if (video.paused) {
      if (video.ended) video.currentTime = 0;
      video.play().catch(() => error('Playback could not start. Try the video’s built-in Play control.'));
    } else video.pause();
  }
  $('play').addEventListener('click', toggle);
  $('restart').addEventListener('click', () => seek(0));
  $('previous').addEventListener('click', () => seek(data.chapters[Math.max(0, sceneIndex() - 1)].start));
  $('next').addEventListener('click', () => seek(data.chapters[Math.min(data.chapters.length - 1, sceneIndex() + 1)].start));
  $('fullscreen').addEventListener('click', () => {
    if (video.requestFullscreen) video.requestFullscreen().catch(() => error('Use the video’s built-in fullscreen control.'));
    else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
  });
  ['timeupdate', 'seeked', 'play', 'pause', 'loadedmetadata'].forEach(event => video.addEventListener(event, update));
  video.addEventListener('loadedmetadata', () => {
    if (Math.abs(video.duration - data.duration) > .15) error('The video and replay data do not match. Regenerate them with python render_presentation.py.');
  });
  video.addEventListener('error', () => error('The animation is missing or cannot be played. Run python render_presentation.py and reopen the demo.'));
  video.addEventListener('ended', () => { $('results').open = true; update(); });
  document.addEventListener('keydown', event => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName)) return;
    if (event.code === 'Space' && !['BUTTON', 'SUMMARY', 'A', 'VIDEO'].includes(event.target.tagName)) { event.preventDefault(); toggle(); }
    if (event.key === 'ArrowLeft') { event.preventDefault(); $('previous').click(); }
    if (event.key === 'ArrowRight') { event.preventDefault(); $('next').click(); }
  });
  update();
})();
