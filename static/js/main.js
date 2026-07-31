// Playback speed for experiment videos (change PLAYBACK_RATE to adjust)
const PLAYBACK_RATE = 3;

document.querySelectorAll('.video-card video, .teaser-video video').forEach((v) => {
  v.playbackRate = PLAYBACK_RATE;
  // re-apply after seeking/reload — some browsers reset the rate
  v.addEventListener('loadedmetadata', () => { v.playbackRate = PLAYBACK_RATE; });
  v.addEventListener('play', () => { v.playbackRate = PLAYBACK_RATE; });
});

// ---- Live phase badge on the teaser video ----
// Phase boundaries in *video-time seconds* (the source clip, not 3x-scaled —
// currentTime already reports source seconds). The teaser clip is ~28s.
// TUNE these to the real phase timings of static/videos/teaser.mp4.
const PHASE_TIMES = [
  { t: 0.0,  key: 'home',              label: 'home' },
  { t: 2.5,  key: 'transport-object',  label: 'transport → object' },
  { t: 5.0, key: 'grasp',             label: 'grasp VLA' },
  { t: 17.0, key: 'transport-basket',  label: 'transport → basket' },
  { t: 21.0, key: 'place',             label: 'place VLA' },
];

(function () {
  const video = document.getElementById('teaser-video');
  const badge = document.getElementById('phase-badge');
  if (!video || !badge) return;
  const textEl = badge.querySelector('.phase-text');

  function currentPhase(time) {
    let cur = PHASE_TIMES[0];
    for (const p of PHASE_TIMES) {
      if (time >= p.t) cur = p; else break;
    }
    return cur;
  }

  function update() {
    const p = currentPhase(video.currentTime);
    if (badge.dataset.phase !== p.key) {
      badge.dataset.phase = p.key;
      textEl.textContent = p.label;
    }
    requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
})();
