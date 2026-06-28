// Граница ошибок: перехват uncaught exceptions, UI с кнопкой сброса.
import React from 'react';

// ── Lifecycle: перехват ошибок React ──
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    console.error('[ErrorBoundary] Caught error:', error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  // ── Render: UI ошибки или дочерние элементы ──
  render() {
    if (this.state.hasError) {
      const { error, errorInfo } = this.state;
      return (
        <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-6">
          <div className="max-w-xl w-full bg-card border border-border rounded-lg p-6 shadow-xl">
            <h1 className="text-xl font-semibold text-red-400 mb-3">
              Произошла ошибка
            </h1>
            <p className="text-sm text-muted-foreground mb-4">
              Приложение столкнулось с непредвиденной ошибкой. Можно попробовать
              перезагрузить страницу или вернуться к выбору блокнота.
            </p>
            {error && (
              <details className="mb-4 text-xs">
                <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                  Технические детали
                </summary>
                <pre className="mt-2 p-3 bg-background/50 rounded overflow-auto text-red-300 whitespace-pre-wrap">
                  {error.toString()}
                  {errorInfo && errorInfo.componentStack ? '\n\nStack:\n' + errorInfo.componentStack : ''}
                </pre>
              </details>
            )}
            <div className="flex gap-2">
              <button
                onClick={this.handleReset}
                className="px-4 py-2 bg-primary text-primary-foreground rounded hover:opacity-90 text-sm"
              >
                Сбросить ошибку
              </button>
              <button
                onClick={this.handleReload}
                className="px-4 py-2 bg-secondary text-secondary-foreground rounded hover:opacity-90 text-sm"
              >
                Перезагрузить страницу
              </button>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
