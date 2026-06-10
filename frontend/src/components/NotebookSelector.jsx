import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Plus, Trash2, Book, Clock } from 'lucide-react';
import axios from 'axios';
import { cn } from '../lib/utils';

export default function NotebookSelector({ onSelect }) {
  const [notebooks, setNotebooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('date_desc'); // 'date_desc', 'date_asc', 'name_asc', 'name_desc'
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newNotebookName, setNewNotebookName] = useState('');

  useEffect(() => {
    fetchNotebooks();
  }, []);

  const fetchNotebooks = async () => {
    try {
      const res = await axios.get('/api/notebooks');
      setNotebooks(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!newNotebookName.trim()) return;
    try {
      await axios.post('/api/notebooks', { name: newNotebookName });
      setNewNotebookName('');
      setIsCreateModalOpen(false);
      fetchNotebooks();
    } catch (err) {
      alert('Ошибка создания');
    }
  };

  const deleteNotebook = async (e, id) => {
    e.stopPropagation();
    if (!confirm('Удалить этот блокнот и все его данные?')) return;
    try {
      await axios.delete(`/api/notebooks/${id}`);
      fetchNotebooks();
    } catch (err) {
      alert('Ошибка удаления');
    }
  };

  const filteredNotebooks = notebooks
    .filter(nb => nb.name.toLowerCase().includes(searchTerm.toLowerCase()))
    .sort((a, b) => {
      switch (sortBy) {
        case 'name_asc': return a.name.localeCompare(b.name);
        case 'name_desc': return b.name.localeCompare(a.name);
        case 'date_asc': return a.created_at - b.created_at;
        case 'date_desc': 
        default: return b.created_at - a.created_at;
      }
    });

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="flex flex-col items-center justify-start min-h-screen p-8 bg-background"
    >
      <div className="text-center mb-12 mt-12">
        <motion.h1 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-6xl font-bold tracking-tighter mb-4"
        >
          DocuMind <span className="text-primary">Local</span>
        </motion.h1>
        <p className="text-muted-foreground text-lg max-w-md mx-auto">
          Ваши персональные ИИ-блокноты. Вся мощь анализа документов в полной приватности.
        </p>
      </div>

      <div className="flex flex-col items-center gap-6 w-full max-w-5xl mb-12">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => setIsCreateModalOpen(true)}
          className="flex items-center gap-2 bg-primary text-primary-foreground px-8 py-3 rounded-xl font-semibold shadow-lg shadow-primary/20 transition-all w-full md:w-auto justify-center"
        >
          <Plus size={20} />
          Создать новый блокнот
        </motion.button>

        <div className="flex flex-col md:flex-row gap-4 w-full items-center bg-card/30 backdrop-blur-md p-2 rounded-2xl border border-border/40 shadow-inner">
          <div className="relative flex-1 w-full flex items-center">
            <Plus size={16} className="absolute left-4 text-muted-foreground/40" />
            <input 
              type="text" 
              placeholder="Поиск по названию..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-transparent border-none outline-none pl-11 pr-4 py-2.5 text-sm font-medium placeholder:text-muted-foreground/30"
            />
          </div>
          <div className="flex items-center gap-1.5 p-1 bg-muted/20 rounded-xl border border-border/20 overflow-x-auto no-scrollbar">
            <span className="text-[10px] font-bold text-muted-foreground/40 px-3 uppercase tracking-tighter whitespace-nowrap">Сортировка:</span>
            {[
              { id: 'date_desc', label: 'Новые' },
              { id: 'date_asc', label: 'Старые' },
              { id: 'name_asc', label: 'А-Я' },
              { id: 'name_desc', label: 'Я-А' },
            ].map(opt => (
              <button 
                key={opt.id}
                onClick={() => setSortBy(opt.id)}
                className={cn(
                  "px-4 py-1.5 text-[11px] font-bold rounded-lg transition-all whitespace-nowrap",
                  sortBy === opt.id 
                    ? "bg-primary text-primary-foreground shadow-md shadow-primary/20 scale-[1.02]" 
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/40"
                )}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 w-full max-w-5xl">
        {filteredNotebooks.map((nb, i) => (
          <motion.div
            key={nb.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * i }}
            onClick={() => onSelect(nb)}
            className="group relative glass-card p-8 rounded-3xl cursor-pointer overflow-hidden border border-border/50 hover:border-primary/30"
          >
            <div className="absolute top-0 right-0 p-4 opacity-0 group-hover:opacity-100 transition-opacity z-10">
              <button 
                onClick={(e) => deleteNotebook(e, nb.id)}
                className="p-2 hover:bg-destructive/10 text-muted-foreground hover:text-destructive rounded-xl transition-colors"
              >
                <Trash2 size={18} />
              </button>
            </div>
            
            <div className="w-12 h-12 bg-primary/10 rounded-2xl flex items-center justify-center text-primary mb-6 group-hover:scale-110 transition-transform">
              <Book size={24} />
            </div>

            <h3 className="text-xl font-bold mb-2 group-hover:text-primary transition-colors">{nb.name}</h3>
            
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Clock size={14} />
              <span>{new Date(nb.created_at * 1000).toLocaleDateString()}</span>
            </div>

            <div className="mt-8 flex items-center text-sm font-medium text-primary opacity-0 group-hover:opacity-100 transition-all transform translate-x-[-10px] group-hover:translate-x-0">
              Открыть блокнот →
            </div>
          </motion.div>
        ))}
      </div>

      {filteredNotebooks.length === 0 && !loading && (
        <div className="mt-12 text-muted-foreground text-center">
          <p>Ничего не найдено</p>
        </div>
      )}

      {loading && <div className="mt-12 animate-pulse text-muted-foreground">Загрузка ваших блокнотов...</div>}

      {/* Модальное окно создания блокнота */}
      {isCreateModalOpen && (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onClick={() => setIsCreateModalOpen(false)}
            className="absolute inset-0 bg-background/80 backdrop-blur-md"
          />
          <motion.div 
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            className="relative bg-card border border-border shadow-2xl rounded-3xl p-8 w-full max-w-md"
          >
            <h2 className="text-2xl font-bold mb-2 text-center">Новый блокнот</h2>
            <p className="text-muted-foreground text-sm text-center mb-8">
              Введите название для вашего нового рабочего пространства.
            </p>
            <form onSubmit={handleCreateSubmit} className="space-y-6">
              <div className="space-y-2">
                <input 
                  autoFocus
                  type="text" 
                  placeholder="Название блокнота" 
                  value={newNotebookName}
                  onChange={(e) => setNewNotebookName(e.target.value)}
                  className="w-full bg-muted/50 border border-border/50 rounded-xl px-4 py-3 outline-none focus:border-primary/50 focus:ring-4 focus:ring-primary/10 transition-all font-medium"
                />
              </div>
              <div className="flex gap-3">
                <button 
                  type="button"
                  onClick={() => setIsCreateModalOpen(false)}
                  className="flex-1 px-4 py-3 rounded-xl border border-border hover:bg-muted transition-colors font-semibold text-sm"
                >
                  Отмена
                </button>
                <button 
                  type="submit"
                  disabled={!newNotebookName.trim()}
                  className="flex-[2] px-4 py-3 rounded-xl bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-50 transition-all font-semibold text-sm shadow-lg shadow-primary/20"
                >
                  Создать
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </motion.div>
  );
}
