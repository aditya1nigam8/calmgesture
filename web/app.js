'use strict';
const $ = id => document.getElementById(id);
const names = {rolling:'Rolling', shaking:'Shaking', tapping:'Tapping', swiping:'Swiping', background:'Resting'};
const frequencies = {rolling:261.63, shaking:329.63, tapping:392, swiping:440};
let generation = 0, stream = null, session = null, timer = null, audio = null, voice = null, muted = false;
const capture = document.createElement('canvas');
const edges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
function silence() {
  if (!voice) return;
  const old = voice; voice = null;
  old.gain.gain.cancelScheduledValues(audio.currentTime);
  old.gain.gain.setTargetAtTime(0, audio.currentTime, .035);
  old.osc.stop(audio.currentTime + .2);
  old.osc.onended = () => { old.osc.disconnect(); old.gain.disconnect(); };
}
function sound(label) {
  if (muted || !frequencies[label] || !audio || audio.state !== 'running') { silence(); return; }
  const volume = Number($('volume').value) / 500;
  if (voice && voice.label === label) { voice.gain.gain.setTargetAtTime(volume, audio.currentTime, .1); return; }
  silence();
  const osc = audio.createOscillator(), gain = audio.createGain();
  osc.type = 'sine'; osc.frequency.value = frequencies[label]; gain.gain.value = 0;
  osc.connect(gain); gain.connect(audio.destination); osc.start();
  gain.gain.setTargetAtTime(volume, audio.currentTime, .15);
  voice = {osc, gain, label};
}
async function request(path, body, token) {
  const response = await fetch(path, {method:'POST', headers: token ? {'X-Session':token} : {}, body, signal:AbortSignal.timeout(5000)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not connect to the local app.');
  return data;
}
function clearView() {
  $('overlay').getContext('2d').clearRect(0,0,$('overlay').width,$('overlay').height);
  document.querySelectorAll('[data-gesture]').forEach(card => card.classList.remove('active'));
  $('progress').value = 0;
}
function stop(message = 'Session ended. You can start again whenever you like.') {
  generation++; clearTimeout(timer); silence();
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null; $('video').srcObject = null;
  const previous = session; session = null;
  if (previous) request('/api/stop', null, previous).catch(() => {});
  $('start').disabled = false; $('stop').disabled = true;
  $('placeholder').hidden = false; $('badge').textContent = 'CAMERA OFF';
  $('gesture').textContent = 'Ready when you are'; $('message').textContent = message;
  $('diagnostics').textContent = 'Waiting for a movement.'; clearView();
}
function render(data) {
  $('progress').max = data.required_samples; $('progress').value = data.samples;
  $('gesture').textContent = data.active ? names[data.active] : 'Listening to your movement';
  $('message').textContent = data.message;
  $('badge').textContent = data.landmarks.length ? 'HAND IN VIEW' : 'LOOKING FOR YOUR HAND';
  $('diagnostics').textContent = `Prediction: ${names[data.prediction] || 'collecting'} · score ${Math.round(data.confidence * 100)}% · processing ${Math.round(data.processing_ms)} ms. Scores are not calibrated probabilities of correctness.`;
  const canvas = $('overlay'), ctx = canvas.getContext('2d');
  canvas.width = capture.width; canvas.height = capture.height;
  ctx.strokeStyle = '#d6e8af'; ctx.fillStyle = '#f5f8df'; ctx.lineWidth = 2;
  if (data.landmarks.length) {
    for (const [a,b] of edges) { ctx.beginPath(); ctx.moveTo(data.landmarks[a][0]*canvas.width,data.landmarks[a][1]*canvas.height); ctx.lineTo(data.landmarks[b][0]*canvas.width,data.landmarks[b][1]*canvas.height); ctx.stroke(); }
    for (const [x,y] of data.landmarks) { ctx.beginPath(); ctx.arc(x*canvas.width,y*canvas.height,3,0,Math.PI*2); ctx.fill(); }
  }
  document.querySelectorAll('[data-gesture]').forEach(card => card.classList.toggle('active',card.dataset.gesture === data.active));
  sound(data.active);
}
async function sample(run) {
  if (run !== generation || !session) return;
  const started = performance.now();
  try {
    const video = $('video'); capture.width = Math.min(640,video.videoWidth);
    capture.height = Math.round(capture.width * video.videoHeight / video.videoWidth);
    capture.getContext('2d').drawImage(video,0,0,capture.width,capture.height);
    const blob = await new Promise(resolve => capture.toBlob(resolve,'image/jpeg',.85));
    if (run !== generation) return;
    // Silence a stalled inference request instead of holding a tone indefinitely.
    const watchdog = setTimeout(silence,350);
    let data;
    try { data = await request('/api/frame',blob,session); } finally { clearTimeout(watchdog); }
    if (run !== generation) return;
    render(data); timer = setTimeout(() => sample(run),Math.max(0,1000/12-(performance.now()-started)));
  } catch (error) { if (run === generation) stop(error.message); }
}
$('start').addEventListener('click', async () => {
  const run = ++generation;
  $('start').disabled = true; $('stop').disabled = false; $('message').textContent = 'Opening your camera…';
  try {
    audio ||= new AudioContext(); await audio.resume();
    if (run !== generation) return;
    const acquired = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'user'},audio:false});
    if (run !== generation) { acquired.getTracks().forEach(track => track.stop()); return; }
    stream = acquired; $('video').srcObject = stream; await $('video').play();
    if (run !== generation) return;
    const result = await request('/api/start');
    if (run !== generation) { request('/api/stop',null,result.session).catch(() => {}); return; }
    session = result.session; $('placeholder').hidden = true;
    stream.getVideoTracks()[0].addEventListener('ended',() => { if(run === generation) stop('Camera disconnected. Please start again.'); });
    sample(run);
  } catch (error) { if (run === generation) stop(error.name === 'NotAllowedError' ? 'Camera permission is needed. Allow it in your browser, then try again.' : error.message); }
});
$('stop').addEventListener('click',() => stop());
$('mute').addEventListener('click',() => { muted = !muted; $('mute').textContent = muted ? 'Unmute sound' : 'Mute sound'; $('mute').setAttribute('aria-pressed',String(muted)); if(muted) silence(); });
$('volume').addEventListener('input',() => { $('level').textContent = `${$('volume').value}%`; if(voice) sound(voice.label); });
document.addEventListener('visibilitychange',() => { if(document.hidden) stop('Session paused while this tab was away. Press Start to continue.'); });
window.addEventListener('pagehide',() => stop());
