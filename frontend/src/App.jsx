// Корневой компонент: переключение между выбором блокнота и основным приложением.
import React, { useState } from 'react';
import NotebookSelector from './components/NotebookSelector';
import MainApp from './components/MainApp';
import ErrorBoundary from './components/ErrorBoundary';
import { AnimatePresence } from 'framer-motion';

function App() {
  const [currentNotebook, setCurrentNotebook] = useState(null);

  return (
    <ErrorBoundary>
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
    </ErrorBoundary>
  );
}

export default App;
