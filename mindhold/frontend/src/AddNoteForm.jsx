import { useState } from "react";
import { createNote } from "./api";

export default function AddNoteForm({ onCreated, onCancel }) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!title.trim() || !description.trim() || saving) return;
    setSaving(true);
    const note = await createNote(title.trim(), description.trim());
    setSaving(false);
    onCreated(note);
  };

  return (
    <form className="add-note-form" onSubmit={handleSubmit}>
      <input
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Title"
        autoFocus
      />
      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Description"
        rows={4}
      />
      <div className="add-note-form-actions">
        <button type="button" onClick={onCancel}>
          Cancel
        </button>
        <button type="submit" disabled={saving || !title.trim() || !description.trim()}>
          {saving ? "Saving..." : "Save"}
        </button>
      </div>
    </form>
  );
}
