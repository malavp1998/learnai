import { useState } from "react";
import Sidebar from "./Sidebar";
import NotesView from "./NotesView";
import ChatView from "./ChatView";
import "./App.css";

export default function App() {
  const [activeView, setActiveView] = useState("notes");

  return (
    <div className="app-shell">
      <Sidebar activeView={activeView} onNavigate={setActiveView} />
      <main className="main-content">
        {activeView === "notes" ? <NotesView /> : <ChatView />}
      </main>
    </div>
  );
}
