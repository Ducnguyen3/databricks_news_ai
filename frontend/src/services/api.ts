import axios from 'axios';
import type { ChatFilters, ChatResponse, CurrentContext } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sendChatMessage = async (
  question: string,
  filters: ChatFilters,
  debug: boolean,
  currentContext?: CurrentContext
): Promise<ChatResponse> => {
  const url = API_BASE_URL.endsWith('/api') ? '/chat' : '/api/chat';
  const response = await api.post<ChatResponse>(url, {
    question,
    filters,
    debug,
    current_context: currentContext,
  });
  return response.data;
};
