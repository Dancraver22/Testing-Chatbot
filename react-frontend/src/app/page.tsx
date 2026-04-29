"use client";

import { useState } from "react";

export default function Home() {
  const [chatHistory, setChatHistory] = useState([
    { role: "assistant", content: "Hello! I am your AI assistant. How can I help you today?" }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const personas = ["Professional", "Sassy", "Emo"];
  const [selectedPersona, setSelectedPersona] = useState(personas[0]);

  const BACKEND_URL = "https://dancraver-rag-prototype.hf.space";

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    const newHistory = [...chatHistory, { role: "user", content: userMessage }];
    setChatHistory(newHistory);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          persona: selectedPersona,
          history: chatHistory.slice(-5), // Send last 5 messages
          user_tz: Intl.DateTimeFormat().resolvedOptions().timeZone,
          image_data: null, // Placeholder for vision
          local_data_context: "No file uploaded." // Placeholder for file upload
        })
      });

      if (response.ok) {
        const data = await response.json();
        setChatHistory([...newHistory, { role: "assistant", content: data.response }]);
      } else {
        setChatHistory([...newHistory, { role: "assistant", content: `Error: ${response.statusText}` }]);
      }
    } catch (error) {
      console.error("Failed to connect to backend:", error);
      setChatHistory([...newHistory, { role: "assistant", content: "Error: Could not connect to the Cloud Agent." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-80 glass-panel border-r border-white/10 flex flex-col h-full z-10 shadow-2xl">
        <div className="p-6 border-b border-white/10">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent flex items-center gap-2">
            <span className="text-3xl">🌐</span> Global Vision AI
          </h1>
        </div>
        
        <div className="p-6 flex-1 flex flex-col gap-6 overflow-y-auto">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-400 uppercase tracking-wider">Persona</label>
            <select 
              value={selectedPersona}
              onChange={(e) => setSelectedPersona(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all appearance-none"
            >
              {personas.map(p => <option key={p} value={p} className="bg-gray-900">{p}</option>)}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-400 uppercase tracking-wider">Context Data</label>
            <div className="border-2 border-dashed border-white/10 rounded-xl p-6 text-center hover:bg-white/5 transition-colors cursor-pointer group">
              <div className="text-3xl mb-2 group-hover:scale-110 transition-transform">📊</div>
              <div className="text-sm text-gray-300 font-medium">Upload CSV/Excel</div>
            </div>
            <button className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 rounded-lg font-medium transition-colors mt-2 text-sm shadow-lg shadow-indigo-500/20">
              Index to Memory
            </button>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-400 uppercase tracking-wider">Vision</label>
            <div className="border-2 border-dashed border-white/10 rounded-xl p-6 text-center hover:bg-white/5 transition-colors cursor-pointer group">
              <div className="text-3xl mb-2 group-hover:scale-110 transition-transform">📸</div>
              <div className="text-sm text-gray-300 font-medium">Upload Image</div>
            </div>
          </div>
        </div>
        
        <div className="p-4 border-t border-white/10">
           <button 
             onClick={() => setChatHistory([])}
             className="w-full py-3 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg text-sm font-medium transition-colors"
           >
             Clear Chat History
           </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative">
        <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl mx-auto w-full">
          {chatHistory.length === 0 ? (
             <div className="h-full flex items-center justify-center text-gray-500">
               <div className="text-center space-y-4">
                 <div className="text-6xl drop-shadow-lg">✨</div>
                 <p className="text-xl">Start a conversation...</p>
               </div>
             </div>
          ) : (
            chatHistory.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-2xl p-5 ${
                  msg.role === 'user' 
                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20 rounded-tr-sm' 
                    : 'glass-panel rounded-tl-sm text-gray-200'
                }`}>
                  {msg.content}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Input Area */}
        <div className="p-6 max-w-4xl mx-auto w-full relative">
          <form 
            onSubmit={handleSendMessage}
            className="glass-panel p-2 rounded-2xl flex items-center gap-2 focus-within:ring-2 focus-within:ring-indigo-500/50 transition-all shadow-xl"
          >
            <input 
              type="text" 
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Ask the Agent..." 
              disabled={isLoading}
              className="flex-1 bg-transparent border-none focus:outline-none px-4 text-white placeholder-gray-500 text-lg disabled:opacity-50"
            />
            <button 
              type="submit"
              disabled={isLoading}
              className={`w-12 h-12 rounded-xl flex items-center justify-center transition-transform hover:scale-105 active:scale-95 shadow-lg shadow-indigo-500/20 ${isLoading ? 'bg-indigo-400' : 'bg-indigo-600 hover:bg-indigo-500'}`}
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              ) : (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M22 2L11 13" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              )}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
