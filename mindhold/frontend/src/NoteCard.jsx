export default function NoteCard({ note }) {
  const date = new Date(note.created_at).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  return (
    <div className="note-card">
      <h3>{note.title}</h3>
      <p>{note.description}</p>
      <div className="note-date">{date}</div>
    </div>
  );
}
