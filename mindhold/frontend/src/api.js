// Thin wrapper for talking to the FastAPI backend.
export const API_BASE = "http://localhost:8000";

export async function listNotes() {
  const response = await fetch(`${API_BASE}/api/notes`);
  return response.json();
}

export async function createNote(title, description) {
  const response = await fetch(`${API_BASE}/api/notes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, description }),
  });
  return response.json();
}
