// Корневой компонент: переключение между выбором блокнота и основным приложением.
// ── Импорты ──
import React, { useState } from 'react';
import NotebookSelector from './components/NotebookSelector';
import MainApp from './components/MainApp';
import ErrorBoundary from './components/ErrorBoundary';
// ── Состояние и обработчики ──
function App() {
  const [currentNotebook, setCurrentNotebook] = useState(null);
  const [transitioning, setTransitioning] = useState(null);

  const handleSelect = (nb) => {
    setTransitioning('open');
    setCurrentNotebook(nb);
  };

  const handleExit = () => {
    setTransitioning('close');
    setCurrentNotebook(null);
  };

  // ── Рендер компонента ──
  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-background text-foreground overflow-hidden">
        {!currentNotebook ? (
          <div className={`transition-opacity duration-300 ${transitioning === 'close' ? 'animate-fadeIn' : ''}`}>
            <NotebookSelector
              onSelect={handleSelect}
            />
          </div>
        ) : (
          <div className={`animate-fadeIn`}>
            <MainApp
              notebook={currentNotebook}
              onExit={handleExit}
            />
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
}

export default App;
