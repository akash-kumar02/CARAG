# CARAG — Corrective & Adaptive Agentic RAG

Responsive dark-mode chatbot frontend with HTML, CSS, and vanilla JavaScript.

## Files
- `index.html` — page structure
- `style.css` — layout, colors, responsiveness, animations
- `script.js` — chat interactions, loading indicator, API integration

## Run
Open `index.html` in a browser. It starts in demo mode and does not need a backend.

## Connect your Python backend
In `script.js`, set `DEMO_MODE = false` and update `API_URL` to your backend endpoint.

The frontend sends `POST` JSON:
```json
{"query":"What is corrective RAG?"}
```
Expected response:
```json
{"answer":"Your grounded answer","sources":[{"title":"document.pdf","score":0.92,"context":"Retrieved passage"}]}
```
Scores are expected in the 0–1 range. Configure CORS on your Python backend if the frontend and backend have different origins. Do not put API keys in frontend JavaScript.
