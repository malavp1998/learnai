import { useEffect, useState } from "react";
import { listNotes } from "./api";
import NoteCard from "./NoteCard";
import AddNoteForm from "./AddNoteForm";

export default function NotesView() {
  const [notes, setNotes] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listNotes().then((data) => {
      setNotes(data);
      setLoading(false);
    });
  }, []);

  const handleCreated = (note) => {
    setNotes((prev) => [note, ...prev]);
    setShowForm(false);
  };

  return (
    <div className="notes-view">
      <div className="notes-header">
        <h1>Notes</h1>
        <button onClick={() => setShowForm((v) => !v)}>
          {showForm ? "Close" : "+ Add Note"}
        </button>
      </div>

      {showForm && (
        <AddNoteForm onCreated={handleCreated} onCancel={() => setShowForm(false)} />
      )}

      {loading ? (
        <p className="empty-hint">Loading notes...</p>
      ) : notes.length === 0 ? (
        <p className="empty-hint">No notes yet. Add one to get started.</p>
      ) : (
        <div className="notes-grid">
          {notes.map((note) => (
            <NoteCard key={note.id} note={note} />
          ))}
        </div>
      )}
    </div>
  );
}
