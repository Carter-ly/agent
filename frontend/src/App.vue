<template>
  <div class="app">
    <!-- 进入前的幕布 -->
    <div id="curtain" :class="{ 'fade-out': !showCurtain }" :style="{ display: curtainDisplay }">
      <div class="bg-photo"></div>
      <div class="bg-shade"></div>
      <div class="bottom-bar">
        <div class="curtain-inner">
          <div class="head">
            <div class="big-logo">🎓</div>
            <div class="titles">
              <h2>留学全周期智能助手</h2>
              <div class="tagline">基于 RAG + Agent 的智能留学规划系统 · 让每一份申请都有据可依</div>
            </div>
          </div>
          <div class="curtain-feats">
            <div class="feat"><b>🧭 智能规划</b> —— 无论你已有明确目标，还是毫无头绪，都能帮你理清并生成定制申请方案。</div>
            <div class="feat"><b>💬 知识问答</b> —— 围绕院校、专业、语言要求、费用、签证等留学全周期问题，给出带来源的可靠回答。</div>
            <div class="feat"><b>📚 RAG 知识库</b> —— 答案基于真实检索的院校资料生成，杜绝凭空编造，并标注来源供你核验。</div>
          </div>
          <button id="enter" @click="enterApp">进入网页</button>
        </div>
      </div>
    </div>

    <!-- 主体 -->
    <header>
      <div class="logo">🎓</div>
      <div>
        <h1>留学全周期智能助手</h1>
        <div class="sub">RAG + Agent · 引导式规划 · 知识库检索增强 · 标注来源</div>
      </div>
      <div class="zoom">
        <button @click="zoomAdj(-10)" title="缩小字体">A−</button>
        <span>{{ zoomPct }}%</span>
        <button @click="zoomAdj(10)" title="放大字体">A+</button>
      </div>
      <button v-if="started" id="reset" @click="resetAll">↺ 重新开始</button>
    </header>

    <div id="chat">
      <div v-for="m in messages" :key="m.id" class="msg" :class="m.role" :id="'msg-' + m.id">
        <div class="avatar">{{ m.role === 'user' ? '🧑' : '🤖' }}</div>
        <div class="bubble" v-html="m.html"></div>
      </div>
    </div>

    <div class="composer-wrap" v-show="showComposer">
      <div class="composer">
        <textarea id="input" v-model="inputText" :placeholder="inputPlaceholder" rows="1" @keydown.enter.exact.prevent="send"></textarea>
        <button class="send" id="send" :disabled="busy" @click="send">发送</button>
      </div>
      <div class="hint">Enter 发送 · Shift+Enter 换行</div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, onMounted } from 'vue'
import { COUNTRY_SCHOOLS, EXAM_CATEGORIES, STEPS } from './data'
import { streamRequest } from './chat'

// ---------- 状态 ----------
const showCurtain = ref(true)
const curtainDisplay = ref('flex')
const started = ref(false)
const busy = ref(false)
const inputText = ref('')
const inputPlaceholder = ref('输入你的问题…')
const chatMode = ref('ask')
const history = []
const messages = ref([])

// 页面缩放（字体大小调节）
const zoomLevel = ref(100)
const zoomPct = computed(() => zoomLevel.value)
function zoomAdj(delta) {
  zoomLevel.value = Math.min(150, Math.max(75, zoomLevel.value + delta))
  document.documentElement.style.zoom = String(zoomLevel.value / 100)
}

// 引导状态机
const category = ref(null)    // null | school | life | exam
const mode = ref(null)        // null | plan | ask
const planType = ref(null)    // null | have | none
const stepIdx = ref(0)
const planInfo = ref({ country: '', school: '', program: '', audience: '' })
const examCat = ref(null)
const examType = ref(null)

const BOT_INTRO = '你好！我是「留学全周期智能助手」。我基于留学知识库（RAG）为你提供留学帮助。请先选择你关心的话题：'

let msgSeq = 0

// ---------- 消息管理 ----------
function addMsg(role, html, opts) {
  const m = { id: ++msgSeq, role, html: html || '', opts: opts || {} }
  messages.value.push(m)
  // 返回数组内的响应式引用（ref 数组元素是代理对象），后续修改 m.html 才能触发 Vue 更新
  const reactive = messages.value[messages.value.length - 1]
  nextTick(() => bindEvents(reactive))
  scrollBottom()
  return reactive
}

function bindEvents(m) {
  const root = document.getElementById('msg-' + m.id)
  if (!root) return
  // 选择卡片
  root.querySelectorAll('[data-action]').forEach(el => {
    el.addEventListener('click', () => {
      const act = el.getAttribute('data-action')
      const val = el.getAttribute('data-val')
      if (actions[act]) actions[act](val)
    })
  })
}

const actions = {
  categorySchool() { category.value = 'school'; mode.value = null; planType.value = null; stepIdx.value = 0; renderFlow() },
  categoryLife() { enterLifeChat() },
  categoryExam() { category.value = 'exam'; examCat.value = null; examType.value = null; renderFlow() },
  categoryBack() { category.value = null; mode.value = null; planType.value = null; stepIdx.value = 0; examCat.value = null; examType.value = null; renderFlow() },
  modePlan() { mode.value = 'plan'; planType.value = null; stepIdx.value = 0; renderFlow() },
  modeAsk() { mode.value = 'ask'; enterAskChat() },
  modeBack() { mode.value = null; renderFlow() },
  planTypeHave() { planType.value = 'have'; stepIdx.value = 0; renderFlow() },
  planTypeNone() { planType.value = 'none'; enterNoneChat() },
  planTypeBack() { planType.value = null; renderFlow() },
  stepPrev() {
    if (stepIdx.value > 0) { stepIdx.value--; renderFlow() }
    else { planType.value = null; renderFlow() }
  },
  stepNext() { stepNextHandler() },
  stepPick(val) { pickStep(val) },
  stepOther(val) { stepOtherHandler(val) },
  stepBackToSummary() { stepIdx.value = STEPS.length - 1; renderFlow() },
  generatePlan() { generatePlan() },
  examCatSelect(cat) { examCat.value = cat; examType.value = null; renderFlow() },
  examCatBack() { examCat.value = null; examType.value = null; renderFlow() },
  examSelect(t) { examType.value = t; enterExamChat() },
}

function scrollBottom() {
  nextTick(() => {
    const el = document.getElementById('chat')
    if (el) el.scrollTop = el.scrollHeight
  })
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[c])
}

function inlineMd(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
}

function renderMarkdown(md) {
  const lines = md.split('\n')
  let html = '<div class="md">'
  let inList = false
  for (const raw of lines) {
    const line = raw.trimEnd()
    if (line.startsWith('### ')) { if (inList) { html += '</ul>'; inList = false } html += '<h3>' + esc(line.slice(4)) + '</h3>' }
    else if (line.startsWith('## ')) { if (inList) { html += '</ul>'; inList = false } html += '<h4>' + esc(line.slice(3)) + '</h4>' }
    else if (line.startsWith('# ')) { if (inList) { html += '</ul>'; inList = false } html += '<div class="md-title">' + esc(line.slice(2)) + '</div>' }
    else if (/^[-*] /.test(line)) { if (!inList) { html += '<ul>'; inList = true } html += '<li>' + inlineMd(line.replace(/^[-*] /, '')) + '</li>' }
    else if (line.trim() === '---') { if (inList) { html += '</ul>'; inList = false } html += '<hr/>' }
    else if (line.trim() === '') { if (inList) { html += '</ul>'; inList = false } }
    else { if (inList) { html += '</ul>'; inList = false } html += '<p>' + inlineMd(line) + '</p>' }
  }
  if (inList) html += '</ul>'
  html += '</div>'
  return html
}

function renderSources(sources) {
  if (!sources || !sources.length) return ''
  return '<div class="sources"><div class="src-label">📚 参考来源：</div>' +
    sources.map(s => '<div class="src-item">' + esc(s) + '</div>').join('') + '</div>'
}

// ---------- 渲染流程 ----------
const showComposer = ref(false)

function renderPathHtml() {
  const segs = []
  if (category.value === 'school') {
    segs.push('🏫 院校申请')
    if (mode.value === 'plan') {
      segs.push('🧭 规划')
      if (planType.value === 'have') {
        segs.push('🗂️ 已有目标')
        for (const k of ['country', 'school', 'program', 'audience']) {
          if (planInfo.value[k]) segs.push(esc(planInfo.value[k]))
        }
      }
    }
  } else if (category.value === 'exam') {
    segs.push('📝 考试')
    if (examCat.value) segs.push(esc(examCat.value))
    if (examType.value) segs.push(esc(examType.value))
  }
  return '<div class="path-line"><b>当前选择：</b>' + segs.join(' › ') + '</div>'
}

function renderFlow() {
  messages.value = []
  showComposer.value = false

  addMsg('bot', esc(BOT_INTRO))

  if (category.value === null) {
    renderCategoryChoice()
  } else {
    addMsg('bot', renderPathHtml())
    if (category.value === 'school') {
      if (mode.value === null) renderModeChoice()
      else if (mode.value === 'plan') {
        if (planType.value === null) renderPlanTypeChoice()
        else if (planType.value === 'have') {
          if (stepIdx.value < STEPS.length) renderStepCard()
          else renderPlanSummary()
        }
      }
    } else if (category.value === 'exam') {
      if (examCat.value === null) renderExamCatChoice()
      else if (examType.value === null) renderExamChoice()
    }
  }

  staggerMsgs()
  scrollBottom()
}

function renderCategoryChoice() {
  addMsg('bot', `
    <div class="onboard">
      <div class="choice-grid">
        <div class="choice-card" data-action="categorySchool">
          <div class="e">🏫</div><div class="t">院校申请</div>
          <div class="d">关于学校、专业、申请规划的问题。可生成定制申请方案或直接提问。</div>
        </div>
        <div class="choice-card" data-action="categoryLife">
          <div class="e">🏠</div><div class="t">留学生活</div>
          <div class="d">签证、租房、打工、就医等出国后的生活问题，直接输入即可。</div>
        </div>
        <div class="choice-card" data-action="categoryExam">
          <div class="e">📝</div><div class="t">考试</div>
          <div class="d">雅思、托福、GRE、GMAT 及各国语言/标化考试，先选考试再提问。</div>
        </div>
      </div>
    </div>`)
}

function renderModeChoice() {
  addMsg('bot', `
    <div class="onboard">
      <div class="choice-grid">
        <div class="choice-card" data-action="modePlan">
          <div class="e">🧭</div><div class="t">规划</div>
          <div class="d">帮你规划留学方案。可选择「已有目标」直接生成定制方案，或「没有目标」由我引导你理清思路。</div>
        </div>
        <div class="choice-card" data-action="modeAsk">
          <div class="e">💬</div><div class="t">提问</div>
          <div class="d">直接问我关于院校、专业、语言要求、费用、签证等任何留学问题。</div>
        </div>
      </div>
      <div class="step-actions"><button class="btn ghost" data-action="categoryBack">← 上一步</button></div>
    </div>`)
}

function renderPlanTypeChoice() {
  addMsg('bot', `
    <div class="onboard">
      <div class="lead">你想怎么规划？</div>
      <div class="choice-grid">
        <div class="choice-card" data-action="planTypeHave">
          <div class="e">🗂️</div><div class="t">已有目标</div>
          <div class="d">我已经想好了国家、学校、专业等目标，请帮我生成一份定制申请方案。</div>
        </div>
        <div class="choice-card" data-action="planTypeNone">
          <div class="e">🧭</div><div class="t">没有目标</div>
          <div class="d">我还没什么思路，请一步步引导我，帮我规划留学方向。</div>
        </div>
      </div>
      <div class="step-actions"><button class="btn ghost" data-action="modeBack">← 上一步</button></div>
    </div>`)
}

function renderStepCard() {
  const step = STEPS[stepIdx.value]
  let chips = step.chips
  if (step.key === 'school') {
    chips = planInfo.value.country ? (COUNTRY_SCHOOLS[planInfo.value.country] || []) : []
  }
  const key = step.key
  const idx = stepIdx.value
  const chipsHtml = chips.map((c, i) =>
    `<span class="chip" data-action="stepPick" data-val="${i}|${key}">${c}</span>`).join('')
    + `<span class="chip" data-action="stepOther" data-val="${key}">其他</span>`

  const m = addMsg('bot', `
    <div class="onboard">
      <div class="step-card">
        <div class="step-title"><span class="n">${idx + 1}</span>${step.title}</div>
        <div class="chips">${chipsHtml}</div>
        <input class="step-input hidden" data-stepinput placeholder="${step.placeholder}" />
        <div class="step-actions">
          <button class="btn ghost" data-action="stepPrev">← 上一步</button>
          <button class="btn" data-action="stepNext">下一步</button>
        </div>
      </div>
    </div>`)
  m.opts = { key, chips }
  nextTick(() => {
    const root = document.getElementById('msg-' + m.id)
    if (!root) return
    const input = root.querySelector('[data-stepinput]')
    const chipEls = root.querySelectorAll('.chip[data-action="stepPick"]')
    if (input) input.addEventListener('input', () => { planInfo.value[key] = input.value.trim() })
    // 回显已有值
    if (planInfo.value[key]) {
      const hit = chips.indexOf(planInfo.value[key])
      if (hit >= 0 && chipEls[hit]) {
        chipEls.forEach(c => c.classList.remove('sel'))
        chipEls[hit].classList.add('sel')
      } else if (input) {
        input.classList.remove('hidden')
        input.value = planInfo.value[key]
        const other = root.querySelector('.chip[data-action="stepOther"]')
        if (other) other.classList.add('sel')
      }
    }
  })
}

function pickStep(val) {
  const [idxStr, key] = val.split('|')
  const value = currentStepChips()[parseInt(idxStr, 10)]
  if (!value) return
  planInfo.value[key] = value
  if (key === 'country' && value) planInfo.value.school = ''
  const root = document.querySelector('#chat .msg:last-child .step-card')
  if (root) {
    root.querySelectorAll('.chip').forEach(c => c.classList.remove('sel'))
    const chips = root.querySelectorAll('.chip[data-action="stepPick"]')
    if (chips[parseInt(idxStr, 10)]) chips[parseInt(idxStr, 10)].classList.add('sel')
    const input = root.querySelector('[data-stepinput]')
    if (input) { input.classList.add('hidden'); input.value = '' }
  }
}

function currentStepChips() {
  const step = STEPS[stepIdx.value]
  if (step.key === 'school') {
    return planInfo.value.country ? (COUNTRY_SCHOOLS[planInfo.value.country] || []) : []
  }
  return step.chips
}

function stepOtherHandler(key) {
  planInfo.value[key] = ''
  if (key === 'country' && planInfo.value.country) planInfo.value.school = ''
  const root = document.querySelector('#chat .msg:last-child .step-card')
  if (root) {
    root.querySelectorAll('.chip').forEach(c => c.classList.remove('sel'))
    const other = root.querySelector('.chip[data-action="stepOther"]')
    if (other) other.classList.add('sel')
    const input = root.querySelector('[data-stepinput]')
    if (input) { input.classList.remove('hidden'); input.value = ''; input.focus() }
  }
}

function stepNextHandler() {
  const step = STEPS[stepIdx.value]
  const val = planInfo.value[step.key]
  if (!val) {
    const input = document.querySelector('#chat .msg:last-child [data-stepinput]')
    if (input) input.focus()
    return
  }
  stepIdx.value++
  renderFlow()
}

function renderPlanSummary() {
  addMsg('bot', `
    <div class="onboard">
      <div class="lead">已确认你的规划目标：</div>
      <div class="summary">
        国家：<b>${esc(planInfo.value.country || '—')}</b> · 学校：<b>${esc(planInfo.value.school || '—')}</b> ·<br/>
        专业：<b>${esc(planInfo.value.program || '—')}</b> · 申请层级：<b>${esc(planInfo.value.audience || '—')}</b>
      </div>
      <div class="step-actions">
        <button class="btn ghost" data-action="stepBackToSummary">← 上一步</button>
        <button class="btn ok" data-action="generatePlan">✨ 生成定制规划方案</button>
      </div>
    </div>`)
}

function renderExamCatChoice() {
  const cards = Object.keys(EXAM_CATEGORIES).map(cat =>
    `<div class="choice-card" data-action="examCatSelect" data-val="${cat}">
       <div class="t">${cat}</div><div class="d">${EXAM_CATEGORIES[cat].desc}</div>
     </div>`).join('')
  addMsg('bot', `
    <div class="onboard">
      <div class="lead">你想咨询哪一类考试？</div>
      <div class="choice-grid">${cards}</div>
      <div class="step-actions"><button class="btn ghost" data-action="categoryBack">← 上一步</button></div>
    </div>`)
}

function renderExamChoice() {
  const catInfo = EXAM_CATEGORIES[examCat.value]
  const cards = catInfo.exams.map(t =>
    `<div class="choice-card" data-action="examSelect" data-val="${t}">
       <div class="t">${t}</div><div class="d">${t} 相关的问题（报名、题型、分数、备考等）</div>
     </div>`).join('')
  addMsg('bot', `
    <div class="onboard">
      <div class="lead">在「${examCat.value}」中，你想咨询哪个考试？</div>
      <div class="choice-grid">${cards}</div>
      <div class="step-actions"><button class="btn ghost" data-action="examCatBack">← 上一步</button></div>
    </div>`)
}

// ---------- 进入对话 ----------
function enterAskChat() {
  addMsg('user', '提问')
  addMsg('bot', '好的！请直接输入你的留学问题，我会检索知识库为你解答。')
  chatMode.value = 'ask'
  showComposer.value = true
  inputPlaceholder.value = '输入你的留学问题，如：去英国留学怎么办理签证？'
  focusInput()
}

function enterNoneChat() {
  addMsg('user', '没有目标')
  addMsg('bot', '好的！请在下方输入你的背景（例如学历、意向国家或专业方向、预算等），我会一步步引导你规划留学方案。')
  chatMode.value = 'advise'
  showComposer.value = true
  inputPlaceholder.value = '输入你的背景，如：我大三，想出国读数据科学硕士，预算有限…'
  focusInput()
}

function enterLifeChat() {
  addMsg('user', '留学生活')
  addMsg('bot', '好的！请输入你的留学生活问题（签证、租房、打工、就医等），我会检索知识库为你解答。')
  chatMode.value = 'ask'
  showComposer.value = true
  inputPlaceholder.value = '输入你的留学生活问题，如：去英国留学怎么办理签证？'
  focusInput()
}

function enterExamChat() {
  addMsg('user', '考试 · ' + examType.value)
  addMsg('bot', `好的！已选择 ${examType.value}。请输入你的问题，我会检索知识库为你解答。`)
  chatMode.value = 'ask'
  showComposer.value = true
  inputPlaceholder.value = `输入关于 ${examType.value} 的问题，如：${examType.value} 总分多少？`
  focusInput()
}

function focusInput() {
  nextTick(() => {
    const el = document.getElementById('input')
    if (el) el.focus()
  })
}

// ---------- 提问 / 引导对话 ----------
async function send() {
  const q = inputText.value.trim()
  if (!q) return
  inputText.value = ''
  await ask(q)
}

async function ask(question) {
  addMsg('user', esc(question))
  const m = addMsg('bot', '<span class="thinking">正在检索知识库并思考…</span><span class="cursor"></span>')
  busy.value = true
  let sources = []
  const endpoint = chatMode.value === 'advise' ? '/api/advise' : '/api/ask'
  try {
    await streamRequest(endpoint, { question, history }, {
      onTool: () => { m.html = '正在检索知识库…<span class="cursor"></span>' },
      onSource: s => { sources = s },
      onToken: (text, answer) => { m.html = esc(answer) + '<span class="cursor"></span>' },
      onDone: (answer) => { m.html = esc(answer) + renderSources(sources) },
    })
    history.push({ role: 'user', content: question })
    history.push({ role: 'assistant', content: m.html.replace(/<[^>]+>/g, '') })
  } catch (err) {
    m.html = '出错了：' + esc(err.message)
  } finally {
    busy.value = false
    focusInput()
  }
  scrollBottom()
}

// ---------- 生成定制方案 ----------
async function generatePlan() {
  const m = addMsg('bot', '<span class="thinking">正在检索知识库并生成你的定制方案…</span><span class="cursor"></span>')
  busy.value = true
  let sources = []
  try {
    await streamRequest('/api/plan', { ...planInfo.value }, {
      onTool: (arg) => { m.html = '🔍 正在检索：' + esc(arg) + '<span class="cursor"></span>' },
      onSource: s => { sources = (sources || []).concat(s || []) },
      onToken: (text, answer) => { m.html = renderMarkdown(answer) + '<span class="cursor"></span>' },
      onDone: (answer) => { m.html = renderMarkdown(answer) + renderSources([...new Set(sources)]) },
    })
  } catch (err) {
    m.html = '出错了：' + esc(err.message)
  } finally {
    busy.value = false
    showComposer.value = true
    inputPlaceholder.value = '方案已生成！继续问我关于申请的任何问题…'
    chatMode.value = 'ask'
    focusInput()
  }
  scrollBottom()
}

// ---------- 重置 / 入场 ----------
function resetAll() {
  history.length = 0
  category.value = null; mode.value = null; planType.value = null; stepIdx.value = 0
  planInfo.value = { country: '', school: '', program: '', audience: '' }
  examCat.value = null; examType.value = null
  chatMode.value = 'ask'
  renderFlow()
}

function enterApp() {
  showCurtain.value = false
  setTimeout(() => {
    curtainDisplay.value = 'none'
    started.value = true
    renderFlow()
    focusInput()
  }, 500)
}

function staggerMsgs() {
  nextTick(() => {
    const msgs = document.querySelectorAll('#chat .msg')
    msgs.forEach((m, i) => {
      m.classList.add('anim')
      m.style.animationDelay = (i * 1000) + 'ms'
    })
  })
}

onMounted(() => {})
</script>

<style>
@import './style.css';

.hidden { display: none !important; }

/* 幕布 */
#curtain {
  position: fixed; inset: 0; z-index: 9999;
  overflow: hidden;
  transition: opacity .5s ease;
  background: #11183c;
}
#curtain.fade-out { opacity: 0; pointer-events: none; }
#curtain .bg-photo {
  position: absolute; inset: 0;
  background: url('/photos/cover-bg.jpg') center/100% 100% no-repeat;
}
#curtain .bg-shade {
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at center, rgba(10,15,40,.22) 0%, rgba(10,15,40,.38) 100%);
}
.bottom-bar {
  position: absolute; left: 0; right: 0; bottom: 0;
  background: rgba(15, 23, 52, .62);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-top: 1px solid rgba(255,255,255,.16);
  padding: 26px 40px 30px;
}
.curtain-inner {
  max-width: 1000px; margin: 0 auto;
  display: flex; flex-direction: column; gap: 16px; align-items: center;
  color: #fff; text-align: center;
}
.curtain-inner .head { display: flex; align-items: center; gap: 14px; }
.curtain-inner .big-logo {
  width: 58px; height: 58px; border-radius: 16px; flex-shrink: 0;
  background: rgba(255,255,255,.14); display: flex; align-items: center; justify-content: center;
  font-size: 30px;
}
.curtain-inner .titles { text-align: left; }
.curtain-inner h2 { font-size: 26px; font-weight: 700; letter-spacing: 1px; }
.curtain-inner .tagline { font-size: 13px; opacity: .85; margin-top: 3px; }
.curtain-feats { display: flex; gap: 12px; text-align: left; }
.curtain-feats .feat {
  flex: 1; min-width: 0;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16);
  border-radius: 10px; padding: 10px 12px; font-size: 12.5px; line-height: 1.5;
}
.curtain-feats .feat b { font-size: 13px; }
#enter {
  background: #fff; color: #11183c; border: none; border-radius: 999px;
  padding: 11px 40px; font-size: 15px; font-weight: 600; cursor: pointer;
  box-shadow: 0 6px 20px rgba(0,0,0,.25); transition: transform .15s;
}
#enter:hover { transform: translateY(-2px) scale(1.03); }

/* 头部 */
header {
  background: var(--card); border-bottom: 1px solid var(--border);
  padding: 12px 20px; display: flex; align-items: center; gap: 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,.05);
  flex-shrink: 0;
}
header .logo {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  display: flex; align-items: center; justify-content: center; color: #fff; font-size: 18px;
}
header h1 { font-size: 18px; font-weight: 600; }
header .sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
.zoom { margin-left: auto; display: flex; align-items: center; gap: 6px; }
.zoom button {
  background: transparent; border: 1px solid var(--border); color: var(--muted);
  font-size: 13px; padding: 5px 9px; cursor: pointer; border-radius: 8px; line-height: 1;
}
.zoom button:hover { color: var(--primary); border-color: var(--primary); }
.zoom span { font-size: 12px; color: var(--muted); min-width: 36px; text-align: center; }
#reset { background: transparent; color: var(--muted); border: 1px solid var(--border); font-size: 13px; padding: 7px 12px; cursor: pointer; }
#reset:hover { color: var(--primary); border-color: var(--primary); }

/* 聊天区 */
#chat {
  flex: 1; min-height: 0; overflow-y: auto; padding: 20px 16px;
  display: flex; flex-direction: column; gap: 16px;
  max-width: 860px; width: 100%; margin: 0 auto;
}
.path-line {
  align-self: center;
  display: inline-block;
  background: rgba(59,130,246,.10); border: 1px solid #c7d2fe;
  color: var(--primary-dark); border-radius: 999px;
  padding: 6px 16px; font-size: 13px;
}
.path-line b { font-weight: 600; }
.msg { display: flex; gap: 10px; max-width: 100%; }
.msg.user { flex-direction: row-reverse; }
.avatar {
  width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center; font-size: 16px;
}
.msg.user .avatar { background: var(--primary); }
.msg.bot .avatar { background: #e0e7ff; }
.bubble {
  padding: 11px 14px; border-radius: 14px; line-height: 1.65;
  max-width: 82%; white-space: pre-wrap; word-break: break-word;
}
.msg.user .bubble { background: var(--user-bubble); color: var(--user-text); border-bottom-right-radius: 4px; }
.msg.bot .bubble { background: var(--bot-bubble); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
.sources { margin-top: 10px; font-size: 12px; }
.sources .src-label { color: var(--muted); margin-bottom: 4px; }
.src-item {
  background: var(--source-bg); border: 1px solid var(--source-border);
  border-radius: 8px; padding: 6px 9px; margin-top: 4px; color: var(--primary-dark);
  word-break: break-all;
}
.thinking { color: var(--muted); font-style: italic; }
.cursor { display: inline-block; width: 8px; height: 16px; background: var(--primary); vertical-align: text-bottom; animation: blink 1s steps(1) infinite; }
@keyframes blink { 50% { opacity: 0; } }

/* Markdown */
.md h3 { font-size: 15px; margin: 12px 0 6px; color: var(--primary-dark); }
.md h4 { font-size: 14px; margin: 10px 0 4px; }
.md ul { padding-left: 20px; margin: 4px 0; }
.md li { margin: 3px 0; }
.md p { margin: 6px 0; }
.md strong { color: var(--text); }
.md code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
.md .md-title { font-size: 17px; font-weight: 700; margin-bottom: 10px; color: var(--text); }
.md hr { border: none; border-top: 1px solid var(--border); margin: 10px 0; }

/* 引导卡片 */
.onboard { display: flex; flex-direction: column; gap: 12px; }
.onboard .lead { font-size: 14px; color: var(--text); margin-bottom: 4px; }
.choice-grid { display: flex; gap: 12px; flex-wrap: wrap; }
.choice-card {
  flex: 1; min-width: 200px; background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 16px; cursor: pointer; text-align: left;
  transition: border-color .15s, box-shadow .15s;
}
.choice-card:hover { border-color: var(--primary); box-shadow: 0 4px 14px rgba(59,130,246,.12); }
.choice-card .t { font-size: 15px; font-weight: 600; margin-bottom: 4px; }
.choice-card .d { font-size: 12px; color: var(--muted); line-height: 1.5; }
.choice-card .e { font-size: 22px; margin-bottom: 6px; }

.step-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 14px;
  padding: 16px; margin-top: 8px;
}
.step-card .step-title { font-size: 14px; font-weight: 600; margin-bottom: 10px; }
.step-card .step-title .n {
  display: inline-block; width: 20px; height: 20px; border-radius: 50%; background: var(--primary);
  color: #fff; font-size: 12px; text-align: center; line-height: 20px; margin-right: 8px;
}
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
.chip {
  background: var(--chip-bg); border: 1px solid var(--chip-border); color: var(--primary-dark);
  border-radius: 999px; padding: 6px 12px; font-size: 13px; cursor: pointer; transition: all .12s;
}
.chip:hover { background: #e0e7ff; }
.chip.sel { background: var(--primary); color: #fff; border-color: var(--primary); }
.step-input { width: 100%; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; font-size: 14px; outline: none; font-family: inherit; }
.step-input:focus { border-color: var(--primary); }
.step-actions { display: flex; justify-content: space-between; gap: 8px; margin-top: 12px; }
.btn { background: var(--primary); color: #fff; border: none; border-radius: 10px; padding: 9px 16px; font-size: 14px; cursor: pointer; font-weight: 500; }
.btn:hover { background: var(--primary-dark); }
.btn:disabled { background: #cbd5e1; cursor: not-allowed; }
.btn.ghost { background: transparent; color: var(--muted); border: 1px solid var(--border); }
.btn.ghost:hover { color: var(--primary); border-color: var(--primary); background: transparent; }
.btn.ok { background: var(--ok); }
.btn.ok:hover { background: #047857; }
.summary { background: var(--source-bg); border: 1px solid var(--source-border); border-radius: 10px; padding: 10px 12px; font-size: 13px; margin-top: 8px; }
.summary b { color: var(--primary-dark); }

/* 输入区 */
.composer-wrap {
  border-top: 1px solid var(--border); background: var(--card); padding: 14px 16px;
}
.composer { max-width: 860px; margin: 0 auto; display: flex; gap: 10px; align-items: flex-end; }
textarea {
  flex: 1; resize: none; border: 1px solid var(--border); border-radius: 12px;
  padding: 12px 14px; font-size: 14px; font-family: inherit; line-height: 1.5;
  min-height: 48px; max-height: 160px; outline: none;
}
textarea:focus { border-color: var(--primary); }
button.send {
  background: var(--primary); color: #fff; border: none; border-radius: 12px;
  padding: 12px 20px; font-size: 14px; cursor: pointer; font-weight: 500;
  transition: background .15s; height: 48px;
}
button.send:hover { background: var(--primary-dark); }
button.send:disabled { background: #cbd5e1; cursor: not-allowed; }
.hint { text-align: center; color: var(--muted); font-size: 12px; padding: 6px 0 0; }

/* 逐条出现动画 */
@keyframes msgIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.msg { animation: msgIn .35s ease both; }
.msg.anim { opacity: 0; animation: msgIn .35s ease forwards; }
</style>