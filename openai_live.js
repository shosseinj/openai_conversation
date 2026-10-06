/* GPT-Live WebRTC only. Local microphone/VAD/verification/Whisper code stays separate.
 * https://developers.openai.com/api/docs/guides/voice-webrtc?api=live
 * https://developers.openai.com/api/docs/guides/live-conversations
 */
class OpenAIVoice {
  constructor() {
    this.state = 'idle'; this.history = []; this.fragments = []; this.seen = new Set();
    this.pc = null; this.dc = null; this.mic = null; this.timer = null;
    this.finalized = false; this.resolveStop = null; this.sessionNumber = 0;
  }
  transcriptMessages() {
    const messages = [];
    const ordered = this.fragments.slice().sort((a,b) => a.session-b.session || a.start-b.start || a.end-b.end || a.order-b.order);
    for (const fragment of ordered) {
      const previous = messages.at(-1);
      if (previous && previous.role === fragment.role && previous.session === fragment.session) previous.text += fragment.text;
      else messages.push({role:fragment.role,text:fragment.text,session:fragment.session});
    }
    return messages;
  }
  startupHistory() {
    const result = []; let bytes = 0;
    for (const message of this.transcriptMessages().slice().reverse()) {
      const size = new TextEncoder().encode(message.text).length;
      if (result.length === 64 || bytes+size > 4096) break;
      if (!message.text.trim()) continue;
      result.unshift({role:message.role,text:message.text}); bytes += size;
    }
    return result;
  }
  renderTranscript() {
    const container = el('openaiTranscript'); container.replaceChildren();
    for (const message of this.transcriptMessages()) {
      const bubble = document.createElement('div'); bubble.className = 'voice-message '+message.role;
      const label = document.createElement('strong'); label.textContent = message.role === 'user' ? 'شما' : 'آوا';
      const content = document.createElement('div'); content.textContent = message.text;
      bubble.append(label,content); container.append(bubble);
    }
    el('output').scrollTop = el('output').scrollHeight;
    if (selectedOpenAI()) {
      el('count').textContent = new Intl.NumberFormat('fa').format(this.transcriptMessages().reduce((n,m)=>n+m.text.trim().split(/\s+/).filter(Boolean).length,0))+' واژه';
      el('copy').disabled = !this.fragments.length;
      el('clear').disabled = this.state !== 'idle' || !this.fragments.length;
    }
  }
  handleEvent(event) {
    if (event.type === 'session.started') {
      this.state = 'listening'; clearTimeout(this.timer);
      el('status').textContent = 'مکالمه فارسی با OpenAI';
      el('hint').textContent = 'صحبت کنید؛ هنگام پاسخ هم می‌توانید صحبت را قطع کنید.';
      refreshEngineControls();
    } else if (event.type === 'session.input_transcript.delta' || event.type === 'session.output_transcript.delta') {
      if (event.event_id && this.seen.has(event.event_id)) return;
      if (typeof event.delta !== 'string' || !Number.isFinite(event.start_ms) || !Number.isFinite(event.end_ms)) return;
      if (event.event_id) this.seen.add(event.event_id);
      this.fragments.push({role:event.type === 'session.input_transcript.delta'?'user':'assistant',
        text:event.delta,start:event.start_ms,end:event.end_ms,session:this.sessionNumber,order:this.fragments.length});
      // Preserve exact deltas, spaces, repetitions and session-relative timestamps.
      this.renderTranscript();
    } else if (event.type === 'session.closed') {
      this.finalized = true;
      this.cleanup(); el('status').textContent = 'مکالمه پایان یافت';
    } else if (event.type === 'error') {
      el('error').textContent = 'خطای OpenAI: '+(event.error?.code || event.code || 'خطا در مکالمه');
      if (this.state === 'connecting') this.cleanup();
    }
  }
  cleanup() {
    clearTimeout(this.timer);
    this.finalized = true; // Suppress recursive close/failure handlers during cleanup.
    this.mic?.getTracks().forEach(track=>track.stop());
    this.dc?.close(); this.pc?.close();
    this.mic = this.dc = this.pc = null;
    el('openaiAudio').pause(); el('openaiAudio').srcObject = null;
    this.state = 'idle';
    if (this.resolveStop) {this.resolveStop(); this.resolveStop=null;}
    refreshEngineControls();
  }
  failure(message) {
    el('openaiApiKey').value = '';
    el('error').textContent = message; this.cleanup();
    el('status').textContent = 'مکالمه متوقف شد';
  }
  async start() {
    if (this.state !== 'idle') return;
    const history = this.startupHistory();
    this.sessionNumber++; this.seen.clear(); this.finalized = false;
    this.state = 'connecting'; el('error').textContent = '';
    el('status').textContent = 'در حال اتصال به OpenAI…'; refreshEngineControls();
    try {
      const pc = this.pc = new RTCPeerConnection();
      pc.ontrack = event => {
        const audio = el('openaiAudio'); audio.srcObject = new MediaStream([event.track]);
        audio.play().catch(()=>{el('hint').textContent='برای شنیدن پاسخ، دکمه پخش صدا را بزنید.';});
      };
      pc.onconnectionstatechange = () => {
        if (!this.finalized && pc.connectionState === 'failed') this.failure('ارتباط صوتی OpenAI برقرار نشد.');
      };
      this.mic = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true},video:false});
      for (const track of this.mic.getAudioTracks()) pc.addTrack(track,this.mic);
      const dc = this.dc = pc.createDataChannel('oai-events');
      dc.onmessage = ({data}) => {try {this.handleEvent(JSON.parse(data));} catch {this.failure('پیام نامعتبر از OpenAI.');}};
      dc.onclose = () => {if (!this.finalized) this.failure('ارتباط OpenAI بدون تأیید پایان جلسه قطع شد.');};
      dc.onerror = () => this.failure('خطا در ارتباط با OpenAI.');
      await pc.setLocalDescription(await pc.createOffer());
      if (pc.iceGatheringState !== 'complete') await new Promise((resolve,reject)=>{
        const timeout=setTimeout(()=>{pc.removeEventListener('icegatheringstatechange',check);reject(Error('مهلت اتصال صوتی پایان یافت.'));},10000);
        function check(){if(pc.iceGatheringState==='complete'){clearTimeout(timeout);pc.removeEventListener('icegatheringstatechange',check);resolve();}}
        pc.addEventListener('icegatheringstatechange',check);check();
      });
      let apiKey = el('openaiApiKey').value.trim();
      el('openaiApiKey').value = '';
      const body = JSON.stringify({sdp:pc.localDescription.sdp,history,api_key:apiKey || undefined});
      apiKey = ''; // The submitted key is never retained in history or browser storage.
      const response = await fetch('/openai-live/session', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body,signal:AbortSignal.timeout(55000)});
      const data = await response.json();
      if (!response.ok) throw Error(data.detail || 'ایجاد مکالمه OpenAI انجام نشد.');
      await pc.setRemoteDescription({type:'answer',sdp:data.transport.sdp});
      // HTTP creates the Live session. Never send Realtime or session.start events.
      if (this.state === 'connecting') this.timer=setTimeout(()=>this.failure('OpenAI آغاز جلسه را تأیید نکرد.'),20000);
    } catch (error) {this.failure(error.name==='NotAllowedError'?'دسترسی به میکروفون را فعال کنید.':error.message);}
  }
  async stop() {
    if (this.state === 'idle') return;
    if (this.state !== 'listening' || this.dc?.readyState !== 'open') {this.cleanup();return;}
    this.state = 'closing'; refreshEngineControls();
    el('status').textContent = 'در حال پایان مکالمه…';
    // Keep media and the event receiver alive until session.closed arrives.
    await new Promise(resolve=>{
      this.resolveStop=resolve;
      this.timer=setTimeout(()=>this.failure('پایان جلسه OpenAI تأیید نشد؛ اتصال بسته شد.'),15000);
      this.dc.send(JSON.stringify({type:'session.close'}));
    });
  }
}

const openaiVoice = window.openaiVoice = new OpenAIVoice();
const selectedOpenAI = () => el('voiceEngine').value === 'openai';
function refreshEngineControls() {
  const live = selectedOpenAI(), active = openaiVoice.state !== 'idle';
  el('voiceEngine').disabled = recording || stopping || enrolling || active;
  el('openaiKeyPanel').hidden = !live;
  el('openaiApiKey').disabled = active;
  document.querySelector('[aria-label="ثبت صدای گوینده"]').hidden = live;
  document.querySelector('.voice-levels').hidden = live;
  el('empty').hidden = el('final').hidden = el('partial').hidden = live;
  el('openaiTranscript').hidden = el('openaiAudio').hidden = !live;
  document.querySelector('footer>span').textContent = live ? 'در این حالت صدای شما برای مکالمه به OpenAI ارسال می‌شود.' : 'صدای شما روی همین دستگاه پردازش می‌شود.';
  if (live) {
    el('toggle').disabled = ['connecting','closing'].includes(openaiVoice.state);
    el('toggle').textContent = active ? 'پایان مکالمه' : 'شروع مکالمه';
    el('orb').classList.toggle('active',active);
    el('toggle').classList.toggle('active',active);
    el('enroll').disabled = active;
    openaiVoice.renderTranscript();
  } else {
    el('toggle').disabled = stopping || enrolling || (!recording && !enrolled);
    if (!recording && !stopping) {
      el('toggle').textContent = 'شروع صحبت';
      el('toggle').classList.remove('active');el('orb').classList.remove('active');
    }
  }
}
window.refreshEngineControls = refreshEngineControls;
const renderBeforeEngines = render;
render = function(){renderBeforeEngines();refreshEngineControls();};
el('toggle').onclick = () => selectedOpenAI() ?
  (openaiVoice.state === 'idle' ? openaiVoice.start() : openaiVoice.stop()) :
  (recording ? stop().catch(error=>fail(error.message)) : start());
el('voiceEngine').onchange = () => {
  if (recording || stopping || enrolling || openaiVoice.state !== 'idle') return;
  if (!selectedOpenAI()) el('openaiApiKey').value = '';
  render();el('error').textContent='';
  el('status').textContent=selectedOpenAI()?'آماده مکالمه با OpenAI':'آماده شنیدن';
  el('hint').textContent=selectedOpenAI()?'برای شروع مکالمه، میکروفون را روشن کنید.':'برای شروع، میکروفون را روشن کنید';
};
const localCopy=el('copy').onclick, localClear=el('clear').onclick;
el('copy').onclick=async()=>{
  if (!selectedOpenAI()) return localCopy();
  try {await navigator.clipboard.writeText(openaiVoice.transcriptMessages().map(m=>(m.role==='user'?'شما: ':'آوا: ')+m.text).join('\n'));}
  catch {el('error').textContent='متن را انتخاب و کپی کنید.';}
};
el('clear').onclick=()=>{
  if (!selectedOpenAI()) return localClear();
  if (openaiVoice.state !== 'idle') return;
  openaiVoice.fragments=[];openaiVoice.seen.clear();openaiVoice.renderTranscript();
};
window.addEventListener('pagehide',()=>{
  el('openaiApiKey').value = '';
  if (openaiVoice.dc?.readyState==='open') openaiVoice.dc.send(JSON.stringify({type:'session.close'}));
  openaiVoice.cleanup();
});
render();
