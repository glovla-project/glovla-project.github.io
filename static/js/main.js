// Playback speed per video group (change to adjust)
const PLAYBACK_RATE = 3;      // teaser + real-robot rollouts
const SIM_PLAYBACK_RATE = 2;  // simulation compare clips

function pinRate(v, rate) {
  v.playbackRate = rate;
  // re-apply after seeking/reload — some browsers reset the rate
  v.addEventListener('loadedmetadata', () => { v.playbackRate = rate; });
  v.addEventListener('play', () => { v.playbackRate = rate; });
}

document.querySelectorAll('.teaser-video video, .video-card video').forEach((v) => {
  const rate = v.closest('.sim-grid') ? SIM_PLAYBACK_RATE : PLAYBACK_RATE;
  pinRate(v, rate);
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
