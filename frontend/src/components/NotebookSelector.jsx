import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Plus, Trash2, Book, Clock } from 'lucide-react';
import axios from 'axios';
import { cn } from '../lib/utils';

export default function NotebookSelector({ onSelect }) {
  const [notebooks, setNotebooks] = useState([]);
  const [loading, setLoading] = useState(true);

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

  const createNotebook = async () => {
    const name = prompt('Введите название блокнота:');
    if (!name) return;
    try {
      await axios.post('/api/notebooks', { name });
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

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="flex flex-col items-center justify-center min-h-screen p-8 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/10 via-background to-background"
    >
      <div className="text-center mb-12">
        <motion.h1 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-6xl font-bold tracking-tighter mb-4"
        >
          NotebookLM <span className="text-primary">Local</span>
        </motion.h1>
        <p className="text-muted-foreground text-lg max-w-md mx-auto">
          Ваши персональные ИИ-блокноты. Вся мощь анализа документов в полной приватности.
        </p>
      </div>

      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={createNotebook}
        className="flex items-center gap-2 bg-primary text-primary-foreground px-8 py-4 rounded-full font-semibold shadow-lg shadow-primary/25 mb-16 transition-all"
      >
        <Plus size={20} />
        Создать новый блокнот
      </motion.button>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 w-full max-w-5xl">
        {notebooks.map((nb, i) => (
          <motion.div
            key={nb.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * i }}
            onClick={() => onSelect(nb)}
            className="group relative glass-card p-8 rounded-3xl cursor-pointer overflow-hidden"
          >
            <div className="absolute top-0 right-0 p-4 opacity-0 group-hover:opacity-100 transition-opacity">
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

            <h3 className="text-xl font-bold mb-2">{nb.name}</h3>
            
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

      {loading && <div className="mt-12 animate-pulse text-muted-foreground">Загрузка ваших блокнотов...</div>}
    </motion.div>
  );
}
