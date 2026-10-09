// SSE 流式请求封装：POST 到 /api/*，逐帧解析 data: {...}
// 事件类型：tool / source / token / done
// 回调 onTool / onSource / onToken / onDone

export async function streamRequest(endpoint, payload, handlers) {
  const { onTool, onSource, onToken, onDone } = handlers || {}
  const resp = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!resp.ok || !resp.body) throw new Error('请求失败 ' + resp.status)

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let answer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const frames = buffer.split('\n\n')
    buffer = frames.pop()
    for (const frame of frames) {
      const line = frame.split('\n').find(l => l.startsWith('data: '))
      if (!line) continue
      const data = JSON.parse(line.slice(6))
      if (data.type === 'tool' && onTool) onTool(data.arg || '')
      else if (data.type === 'source' && onSource) onSource(data.sources || [])
      else if (data.type === 'token' && onToken) { answer += data.text; onToken(data.text, answer) }
      else if (data.type === 'done' && onDone) { answer = data.answer; onDone(data.answer, answer) }
    }
  }
  return answer
}