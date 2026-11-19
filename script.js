const chatMessages = document.getElementById('chatMessages');
const chatContainer = document.getElementById('chatContainer');
const queryInput = document.getElementById('queryInput');
const chatForm = document.getElementById('chatForm');
const sendBtn = document.getElementById('sendBtn');
const welcome = document.getElementById('welcome');

// Set false after connecting your Python FastAPI/Flask backend.
const DEMO_MODE = false;
const API_URL = 'http://localhost:8000/chat';
let isLoading = false;

document.getElementById('statusText').textContent = DEMO_MODE ? 'Demo Mode' : 'Backend Configured';
function scrollToBottom() { chatContainer.scrollTop = chatContainer.scrollHeight; }

function createMessage(role, content, sources = []) {
  const message = document.createElement('div');
  message.className = `message ${role}`;
  const body = document.createElement('div');
  body.className = 'message-body';
  const sender = document.createElement('div');
  sender.className = 'sender';
  sender.textContent = role === 'user' ? 'You' : 'CARAG AI';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const answer = document.createElement('div');
  if (
  role === 'ai' &&
  typeof marked !== 'undefined' &&
  typeof DOMPurify !== 'undefined'
) {
  const rawHTML = marked.parse(content, {
    gfm: true,
    breaks: true
  });

  answer.innerHTML = DOMPurify.sanitize(rawHTML);
} else {
  answer.textContent = content;
}
  bubble.appendChild(answer);

  if (role === 'ai' && sources.length) {
    const section = document.createElement('details');
section.className = 'sources';

const summary = document.createElement('summary');
summary.className = 'sources-toggle';
summary.textContent = `📚 View Sources (${sources.length})`;

section.appendChild(summary);
    const title = document.createElement('div');
    title.className = 'sources-title';
    title.textContent = '▤ Retrieved Sources / Context';
    section.appendChild(title);
    sources.forEach((source, index) => {
      const item = document.createElement('div');
      item.className = 'source-item';
      const name = document.createElement('div');
      name.className = 'source-name';
      name.textContent = `${index + 1}. ${source.title || 'Untitled Source'}`;
      item.appendChild(name);
      if (
  source.page !== undefined &&
  source.page !== null &&
  source.page !== 'N/A'
) {
  const page = document.createElement('div');
  page.className = 'source-page';
  page.textContent = `PDF Page: ${source.page}`;
  item.appendChild(page);
}
      if (source.score !== null && source.score !== undefined && Number.isFinite(Number(source.score))) {
        const score = document.createElement('div');
        score.className = 'source-score';
        score.textContent = `Relevance: ${(Number(source.score) * 100).toFixed(1)}%`;
        item.appendChild(score);
      }
      if (source.context) {
        const context = document.createElement('div');
        context.className = 'source-context';
        context.textContent = source.context;
        item.appendChild(context);
      }
      section.appendChild(item);
    });
    bubble.appendChild(section);
  }
  body.append(sender, bubble);
  if (role === 'ai') {
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '✦';
    message.appendChild(avatar);
  }
  message.appendChild(body);
  chatMessages.appendChild(message);
  scrollToBottom();
}

function showTyping() {
  const message = document.createElement('div');
  message.className = 'message ai';
  message.id = 'typingIndicator';
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = '✦';
  const body = document.createElement('div');
  body.className = 'message-body';
  const sender = document.createElement('div');
  sender.className = 'sender';
  sender.textContent = 'CARAG is thinking...';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.setAttribute('role', 'status');
  typing.setAttribute('aria-label', 'Generating response');
  for (let i = 0; i < 3; i++) typing.appendChild(document.createElement('span'));
  bubble.appendChild(typing);
  body.append(sender, bubble);
  message.append(avatar, body);
  chatMessages.appendChild(message);
  scrollToBottom();
}
function hideTyping() { document.getElementById('typingIndicator')?.remove(); }
function setLoading(value) {
  isLoading = value;
  sendBtn.disabled = value;
  queryInput.disabled = value;


  sendBtn.textContent = value ? 'Thinking...' : 'Send ➤';
}
function getDemoResponse(query) {
  return {
    answer: `This is a demonstration response to your query:\n\n"${query}"\n\nIn the production CARAG system, your question will be analyzed by an agentic RAG pipeline. Relevant documents will be retrieved, evaluated, and used to generate a grounded answer. The corrective component can trigger additional retrieval when evidence is insufficient.`,
    sources: [
      { title: 'Sample Research Document.pdf', score: 0.94, context: 'Illustrative context only. Replace with real retrieved passages from your backend.' },
      { title: 'Sample Knowledge Base', score: 0.87, context: 'Example retrieval result for testing the UI.' }
    ]
  };
}

async function fetchCARAGResponse(query) {
  if (DEMO_MODE) {
    await new Promise(resolve => setTimeout(resolve, 1300));
    return getDemoResponse(query);
  }
  // Replace endpoint/payload as needed for your Python backend.
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  });
  if (!response.ok) {
  if (response.status === 429) {
    throw new Error(
      'Groq API rate limit reached. Please wait a minute and try again.'
    );
  }

  throw new Error(
    `Backend request failed: HTTP ${response.status}`
  );
}
  const data = await response.json();
  // Expected: { answer: string, sources: [{title, score, context}] }
  if (typeof data.answer !== 'string') throw new Error('Invalid API response: missing answer');
  return { answer: data.answer, sources: Array.isArray(data.sources) ? data.sources : [] };
}

async function sendMessage() {
  const query = queryInput.value.trim();
  if (!query || isLoading) return;
  welcome?.remove();
  createMessage('user', query);
  queryInput.value = '';
  queryInput.style.height = '42px';
  setLoading(true);
  showTyping();
  try {
    const result = await fetchCARAGResponse(query);
    hideTyping();
    createMessage('ai', result.answer, result.sources);
  } catch (error) {
    console.error('CARAG Error:', error);
    hideTyping();
    createMessage('ai', error.message);
  } finally {
    setLoading(false);
    queryInput.focus();
  }
}
chatForm.addEventListener('submit', event => { event.preventDefault(); sendMessage(); });
queryInput.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendMessage();
  }
});
queryInput.addEventListener('input', function () {
  this.style.height = '42px';
  this.style.height = `${Math.min(this.scrollHeight, 130)}px`;
});
