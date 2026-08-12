export default function Sidebar({ activeView, onNavigate }) {
  return (
    <div className="sidebar">
      <div className="sidebar-title">mindhold</div>
      <nav className="sidebar-nav">
        <button
          className={activeView === "notes" ? "active" : ""}
          onClick={() => onNavigate("notes")}
        >
          Notes
        </button>
        <button
          className={activeView === "chat" ? "active" : ""}
          onClick={() => onNavigate("chat")}
        >
          Chat
        </button>
      </nav>
    </div>
  );
}
