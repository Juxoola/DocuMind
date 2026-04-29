import React, { useState, useEffect } from 'react';
import NotebookSelector from './components/NotebookSelector';
import MainApp from './components/MainApp';
import { AnimatePresence } from 'framer-motion';

function App() {
  const [currentNotebook, setCurrentNotebook] = useState(null);

  return (
    <div className="min-h-screen bg-background text-foreground overflow-hidden">
      <AnimatePresence mode="wait">
        {!currentNotebook ? (
          <NotebookSelector 
            key="selector"
            onSelect={(nb) => setCurrentNotebook(nb)} 
          />
        ) : (
          <MainApp 
            key="app"
            notebook={currentNotebook} 
            onExit={() => setCurrentNotebook(null)} 
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
