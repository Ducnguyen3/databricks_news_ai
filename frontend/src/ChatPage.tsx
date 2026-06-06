import React, { useMemo, useState } from 'react';
import type { ChatFilters, ChatResponse, CurrentContext, ImageContext, SourceContext } from './types';
import { sendChatMessage } from './services/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

type ChatMessage =
  | { id: string; role: 'user'; text: string; createdAt: Date }
  | { id: string; role: 'ai'; text: string; createdAt: Date; response: ChatResponse };

const QUICK_PROMPTS = [
  'Tin AI gần đây có gì nổi bật?',
  'Tổng hợp tin tức chứng khoán tuần này',
  'Tin tức vào ngày 1 tháng 6 có gì nổi bật?',
  'Có ảnh nào liên quan đến Ukraine không?',
];



export default function ChatPage() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [debugMode, setDebugMode] = useState(false);
  const [selectedSource, setSelectedSource] = useState<SourceContext | null>(null);
  const [hiddenImages, setHiddenImages] = useState<Record<string, boolean>>({});
  const [filters, setFilters] = useState<ChatFilters>({
    time_range_days: 14,
    sources: [],
    topic: '',
    domain: '',
    ticker: '',
    top_k: 5,
  });

  const latestResponse = [...messages].reverse().find((message): message is Extract<ChatMessage, { role: 'ai' }> => message.role === 'ai')?.response;
  const galleryItems = useMemo(() => buildGalleryItems(latestResponse), [latestResponse]);
  const runtime = latestResponse?.debug;

  const submitQuestion = async (text = question) => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    const currentContext = buildCurrentContext({
      previousQuestion: lastUserQuestion(messages),
      previousResponse: latestResponse,
      selectedSource,
      filters,
    });
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', text: trimmed, createdAt: new Date() };

    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setLoading(true);
    setError(null);
    setSelectedSource(null);

    try {
      const response = await sendChatMessage(trimmed, filters, debugMode, currentContext);
      const aiMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'ai',
        text: response.answer || 'Backend không trả về nội dung trả lời.',
        response,
        createdAt: new Date(),
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (err: unknown) {
      const apiError = err as { response?: { status?: number; statusText?: string; data?: { message?: string } } };
      const message = apiError.response
        ? `Backend trả lỗi ${apiError.response.status}: ${apiError.response.data?.message || apiError.response.statusText || 'không rõ lý do'}`
        : `Không kết nối được FastAPI tại ${API_BASE_URL}/api/chat. Kiểm tra backend đang chạy port 8000.`;
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    void submitQuestion();
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
    setSelectedSource(null);
    setHiddenImages({});
  };

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#060910] text-slate-100">
      <Header
        filters={filters}
        debugMode={debugMode}
        runtime={runtime}
        onDebugChange={setDebugMode}
        onFiltersChange={setFilters}
        onClear={clearChat}
      />

      <main className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[32%_1fr]">
        <aside className="hidden min-h-0 border-r border-white/10 bg-[#060910] lg:flex lg:flex-col">
          <Gallery items={galleryItems} />
        </aside>

        <section className="flex min-h-0 flex-col bg-[#0d1117]">
          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-7">
            {messages.length === 0 ? (
              <WelcomeState onPrompt={(prompt) => void submitQuestion(prompt)} />
            ) : (
              <div className="mx-auto flex max-w-4xl flex-col gap-5">
                {messages.map((message) => (
                  <MessageBubble
                    key={message.id}
                    message={message}
                    selectedSource={selectedSource}
                    hiddenImages={hiddenImages}
                    debugMode={debugMode}
                    onSelectSource={setSelectedSource}
                    onImageError={(url) => setHiddenImages((prev) => ({ ...prev, [url]: true }))}
                  />
                ))}
                {loading && <TypingIndicator />}
                {error && <ErrorNotice message={error} />}
              </div>
            )}
          </div>

          <Composer
            question={question}
            loading={loading}
            filters={filters}
            onQuestionChange={setQuestion}
            onSubmit={handleSubmit}
            onPrompt={(prompt) => void submitQuestion(prompt)}
          />
        </section>
      </main>
    </div>
  );
}

function Header({
  filters,
  debugMode,
  runtime,
  onDebugChange,
  onFiltersChange,
  onClear,
}: {
  filters: ChatFilters;
  debugMode: boolean;
  runtime?: ChatResponse['debug'];
  onDebugChange: (value: boolean) => void;
  onFiltersChange: React.Dispatch<React.SetStateAction<ChatFilters>>;
  onClear: () => void;
}) {
  return (
    <header className="flex flex-wrap items-center gap-3 border-b border-white/10 bg-[#060910]/95 px-4 py-3 backdrop-blur sm:px-6">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-300 to-indigo-400 text-sm font-black text-slate-950">
          AI
        </div>
        <div>
          <div className="text-lg font-black tracking-tight">
            News<span className="text-cyan-300">AI</span>
          </div>
          <div className="text-[11px] font-medium text-slate-500">FastAPI RAG · {API_BASE_URL.replace(/^https?:\/\//, '')}</div>
        </div>
      </div>

      <div className="flex flex-1 flex-wrap items-center gap-2">
        <SelectShell label="Bài liên quan">
          <select
            className="news-select"
            value={filters.top_k}
            onChange={(event) => onFiltersChange((prev) => ({ ...prev, top_k: Number(event.target.value) }))}
          >
            <option value={3}>3 bài</option>
            <option value={5}>5 bài</option>
            <option value={10}>10 bài</option>
            <option value={20}>20 bài</option>
          </select>
        </SelectShell>

        <SelectShell label="Thời gian">
          <select
            className="news-select"
            value={filters.time_range_days}
            onChange={(event) => onFiltersChange((prev) => ({ ...prev, time_range_days: Number(event.target.value) }))}
          >
            <option value={1}>Hôm nay</option>
            <option value={7}>Tuần này</option>
            <option value={30}>Tháng này</option>
            <option value={0}>Tất cả</option>
          </select>
        </SelectShell>

      </div>

      <div className="ml-auto flex items-center gap-2">
        {runtime?.chroma_path && (
          <span className="hidden rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] text-slate-500 md:inline">
            {String(runtime.chroma_path).split(/[\\/]/).pop()}
          </span>
        )}
        <label className="flex h-9 items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 text-[11px] font-bold text-slate-400">
          <input type="checkbox" className="accent-cyan-300" checked={debugMode} onChange={(event) => onDebugChange(event.target.checked)} />
          Debug
        </label>
        <button className="h-9 rounded-lg border border-white/10 bg-white/5 px-3 text-[11px] font-bold text-slate-400 hover:text-slate-100" onClick={onClear}>
          Xóa
        </button>
      </div>
    </header>
  );
}

function SelectShell({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wide text-slate-500">
      <span className="hidden sm:inline">{label}</span>
      {children}
    </label>
  );
}

function Gallery({ items }: { items: GalleryItem[] }) {
  return (
    <div className="min-h-0 overflow-y-auto px-4 py-4">
      <div className="mb-4 flex items-center justify-between border-b border-white/10 pb-3">
        <span className="text-[10px] font-bold uppercase tracking-[0.25em] text-slate-500">Tin liên quan</span>
        <span className="flex items-center gap-2 text-[10px] font-bold text-emerald-300">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-300" />
          LIVE
        </span>
      </div>

      <div className="flex flex-col gap-3">
        {items.length ? (
          items.map((item, index) => <GalleryCard key={`${item.title}-${index}`} item={item} />)
        ) : (
          <DefaultInsightCards />
        )}
      </div>
    </div>
  );
}

function GalleryCard({ item }: { item: GalleryItem }) {
  const url = externalUrl(item.url);
  const content = (
    <>
      <div className="relative h-36 overflow-hidden bg-[#141b24]">
        {item.imageUrl ? (
          <img src={item.imageUrl} alt={item.title} className="h-full w-full object-cover opacity-90 transition group-hover:scale-[1.03]" />
        ) : (
          <MiniChart tone={item.tone} />
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-[#060910] via-[#060910]/30 to-transparent" />
      </div>
      <div className="p-3">
        <div className="mb-2 flex items-center justify-between gap-2">
          <span className="rounded bg-white/10 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide text-slate-400">{item.source || 'NewsAI'}</span>
          <span className="text-[10px] text-slate-600">{formatDate(item.publishedAt)}</span>
        </div>
        <h3 className="line-clamp-2 text-sm font-bold leading-5 text-slate-100">{item.title}</h3>
        <span className={`mt-3 inline-flex rounded border px-2 py-0.5 text-[10px] font-bold ${toneClass(item.tone)}`}>{item.label}</span>
      </div>
    </>
  );
  const className = "group overflow-hidden rounded-2xl border border-white/10 bg-[#0d1117] transition hover:-translate-y-0.5 hover:border-white/20";
  if (!url) {
    return <div className={className}>{content}</div>;
  }
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className={className}
    >
      {content}
    </a>
  );
}

function MessageBubble({
  message,
  selectedSource,
  hiddenImages,
  debugMode,
  onSelectSource,
  onImageError,
}: {
  message: ChatMessage;
  selectedSource: SourceContext | null;
  hiddenImages: Record<string, boolean>;
  debugMode: boolean;
  onSelectSource: (source: SourceContext) => void;
  onImageError: (imageUrl: string) => void;
}) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end gap-3">
        <div>
          <div className="max-w-3xl rounded-3xl rounded-br-md border border-white/10 bg-[#1c2533] px-5 py-3 text-sm leading-7 text-slate-100">
            {message.text}
          </div>
          <div className="mt-1 pr-2 text-right text-[10px] text-slate-600">{formatTime(message.createdAt)}</div>
        </div>
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-[#1c2533] text-xs font-bold text-slate-500">U</div>
      </div>
    );
  }

  const response = message.response;
  return (
    <div className="flex gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-300 to-indigo-400 text-xs font-black text-slate-950">
        AI
      </div>
      <div className="min-w-0 flex-1">
        <div className="max-w-4xl rounded-3xl rounded-bl-md border border-white/10 bg-[#141b24] px-5 py-4 text-sm leading-7 text-slate-100">
          <FormattedAnswer text={message.text} />
        </div>
        <ResponseMeta response={response} />
        {response.sources?.length > 0 && <SourceChips sources={response.sources} selectedSource={selectedSource} onSelectSource={onSelectSource} />}
        {response.images?.length > 0 && <InlineImages images={response.images} hiddenImages={hiddenImages} onImageError={onImageError} />}
        {response.generated_image_prompts?.length ? <GeneratedImagePrompts prompts={response.generated_image_prompts} /> : null}
        {response.related_articles?.length ? <RelatedArticles articles={response.related_articles} /> : null}
        {debugMode && <DebugBlock response={response} />}
        <div className="mt-2 text-[10px] text-slate-600">{formatTime(message.createdAt)} · NewsAI</div>
      </div>
    </div>
  );
}

function ResponseMeta({ response }: { response: ChatResponse }) {
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {response.intent && <MetaPill label={response.intent} />}
      {response.topic && <MetaPill label={response.topic} />}
      {response.ticker && <MetaPill label={response.ticker} />}
      {response.query_plan?.time_range_days !== undefined && <MetaPill label={`${response.query_plan.time_range_days} ngày`} />}
    </div>
  );
}

function MetaPill({ label }: { label: string }) {
  return <span className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-2.5 py-1 text-[10px] font-bold text-cyan-200">{label}</span>;
}

function SourceChips({
  sources,
  selectedSource,
  onSelectSource,
}: {
  sources: ChatResponse['sources'];
  selectedSource: SourceContext | null;
  onSelectSource: (source: SourceContext) => void;
}) {
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {sources.map((source) => {
        const selected = selectedSource?.citation_id === source.citation_id;
        const url = externalUrl(source.url);
        const className = `rounded-lg border px-3 py-2 text-left text-[11px] transition ${
          selected ? 'border-cyan-300 bg-cyan-300/10 text-cyan-100' : 'border-white/10 bg-[#1c2533] text-slate-400 hover:border-cyan-300/60'
        }`;
        const content = (
          <>
            <span className="mr-1 font-black text-cyan-300">[{source.citation_id}]</span>
            <span className="font-bold uppercase">{source.source}</span>
            <span className="ml-2 line-clamp-1 max-w-72 text-slate-300">{source.title}</span>
          </>
        );
        if (url) {
          return (
            <a
              key={`${source.citation_id}-${source.article_id}`}
              href={url}
              target="_blank"
              rel="noreferrer"
              className={className}
              title={url}
            >
              {content}
            </a>
          );
        }
        return (
          <button
            key={`${source.citation_id}-${source.article_id}`}
            type="button"
            className={className}
            onClick={() => onSelectSource(source)}
          >
            {content}
          </button>
        );
      })}
    </div>
  );
}

function InlineImages({
  images,
  hiddenImages,
  onImageError,
}: {
  images: ChatResponse['images'];
  hiddenImages: Record<string, boolean>;
  onImageError: (imageUrl: string) => void;
}) {
  const visible = images.filter((image) => {
    const url = imageUrlOf(image);
    return url && !hiddenImages[url];
  });
  if (!visible.length) return null;
  return (
    <div className="mt-4 grid gap-3 sm:grid-cols-2">
      {visible.slice(0, 4).map((image, index) => {
        const url = imageUrlOf(image);
        return (
          <figure key={`${url}-${index}`} className="overflow-hidden rounded-2xl border border-white/10 bg-[#0d1117]">
            <img src={url} alt={image.alt || image.caption || 'Ảnh bài báo'} className="h-44 w-full object-cover" onError={() => onImageError(url)} />
            <figcaption className="space-y-1 p-3 text-xs leading-5 text-slate-400">
              {image.caption && <div>{image.caption}</div>}
              <div className="text-[11px] text-slate-500">
                Nguồn ảnh: <span className="font-bold uppercase text-slate-300">{image.source || 'bài báo'}</span>
                {image.credit ? <span> · {image.credit}</span> : null}
              </div>
              {image.article_title && (
                <a href={image.article_url || undefined} target="_blank" rel="noreferrer" className="block text-[11px] text-cyan-200 hover:text-cyan-100">
                  Bài liên quan: {image.article_title}
                </a>
              )}
            </figcaption>
          </figure>
        );
      })}
    </div>
  );
}

function GeneratedImagePrompts({ prompts }: { prompts: NonNullable<ChatResponse['generated_image_prompts']> }) {
  return (
    <div className="mt-4 rounded-2xl border border-amber-300/20 bg-amber-300/5 p-3">
      <div className="mb-2 text-[10px] font-black uppercase tracking-wider text-amber-200">Prompt minh họa cho bài chưa có ảnh</div>
      <div className="grid gap-2">
        {prompts.slice(0, 4).map((item) => (
          <div key={item.article_id} className="rounded-xl border border-white/10 bg-black/20 p-3">
            <div className="mb-1 text-[11px] font-bold uppercase text-slate-500">{item.source || 'NewsAI'}</div>
            <div className="mb-2 text-xs font-bold text-slate-200">{item.article_title}</div>
            <p className="text-xs leading-5 text-slate-400">{item.prompt}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function RelatedArticles({ articles }: { articles: NonNullable<ChatResponse['related_articles']> }) {
  return (
    <div className="mt-4 grid gap-2 md:grid-cols-2">
      {articles.slice(0, 4).map((article, index) => {
        const url = externalUrl(article.url);
        const content = (
          <>
            <span className="mr-2 font-bold uppercase text-slate-500">{article.source}</span>
            {article.title}
          </>
        );
        const className = "rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-400 hover:border-white/20 hover:text-slate-100";
        if (!url) {
          return (
            <div key={`${article.title}-${index}`} className={className}>
              {content}
            </div>
          );
        }
        return (
          <a
            key={`${url}-${index}`}
            href={url}
            target="_blank"
            rel="noreferrer"
            className={className}
            title={url}
          >
            {content}
          </a>
        );
      })}
    </div>
  );
}

function DebugBlock({ response }: { response: ChatResponse }) {
  return (
    <details className="mt-4 rounded-xl border border-white/10 bg-black/30 p-3">
      <summary className="cursor-pointer text-xs font-bold text-slate-400">Debug retrieval</summary>
      <pre className="mt-3 max-h-80 overflow-auto text-[11px] leading-5 text-slate-300">{JSON.stringify(response, null, 2)}</pre>
    </details>
  );
}

function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-300 to-indigo-400 text-xs font-black text-slate-950">
        AI
      </div>
      <div className="flex w-fit items-center gap-1 rounded-3xl rounded-bl-md border border-white/10 bg-[#141b24] px-5 py-4">
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-500" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-500 [animation-delay:150ms]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-500 [animation-delay:300ms]" />
      </div>
    </div>
  );
}

function ErrorNotice({ message }: { message: string }) {
  return <div className="rounded-2xl border border-red-400/30 bg-red-400/10 px-4 py-3 text-sm text-red-200">{message}</div>;
}

function WelcomeState({ onPrompt }: { onPrompt: (prompt: string) => void }) {
  return (
    <div className="mx-auto flex min-h-full max-w-xl flex-col items-center justify-center py-10 text-center">
      <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-3xl bg-gradient-to-br from-cyan-300 to-indigo-400 text-xl font-black text-slate-950">
        AI
      </div>
      <h1 className="text-2xl font-black tracking-tight text-white">Hỏi bất kỳ điều gì về tin tức</h1>
      <p className="mt-3 text-sm leading-7 text-slate-500">
        Frontend này gửi câu hỏi tới FastAPI `/api/chat`, nhận structured response từ RAG rồi hiển thị câu trả lời, nguồn và ảnh.
      </p>
      <div className="mt-6 grid w-full gap-2">
        {QUICK_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-left text-sm font-bold text-slate-400 transition hover:border-cyan-300/50 hover:bg-cyan-300/10 hover:text-slate-100"
            onClick={() => onPrompt(prompt)}
          >
            <span className="mr-2 text-cyan-300">→</span>
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

function Composer({
  question,
  loading,
  filters,
  onQuestionChange,
  onSubmit,
  onPrompt,
}: {
  question: string;
  loading: boolean;
  filters: ChatFilters;
  onQuestionChange: (value: string) => void;
  onSubmit: (event: React.FormEvent) => void;
  onPrompt: (prompt: string) => void;
}) {
  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <form onSubmit={onSubmit} className="border-t border-white/10 bg-[#0d1117] px-4 py-4 sm:px-6">
      <div className="mx-auto max-w-4xl">
        <div className="mb-3 flex flex-wrap gap-2">
          {QUICK_PROMPTS.slice(0, 3).map((prompt) => (
            <button
              key={prompt}
              type="button"
              className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-[11px] font-bold text-slate-500 hover:border-indigo-300/60 hover:text-indigo-200"
              onClick={() => onPrompt(prompt)}
            >
              {prompt}
            </button>
          ))}
        </div>
        <div className="flex items-end gap-3 rounded-[28px] border border-white/15 bg-[#141b24] px-4 py-3 shadow-[0_0_30px_rgba(0,229,255,0.06)] focus-within:border-cyan-300/50">
          <textarea
            rows={1}
            value={question}
            onChange={(event) => onQuestionChange(event.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            placeholder="Nhập câu hỏi tại đây..."
            className="max-h-32 min-h-10 flex-1 resize-none bg-transparent py-2 text-sm leading-6 text-slate-100 outline-none placeholder:text-slate-600"
          />
          <div className="hidden pb-2 text-[10px] text-slate-600 sm:block">
            {filters.time_range_days ? `${filters.time_range_days} ngày` : 'Tất cả'} · {filters.top_k} bài
          </div>
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-300 to-indigo-400 text-lg font-black text-slate-950 transition hover:scale-105 disabled:cursor-not-allowed disabled:opacity-50"
            title="Gửi"
          >
            →
          </button>
        </div>
      </div>
    </form>
  );
}

function FormattedAnswer({ text }: { text: string }) {
  const html = escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');
  return <div className="prose prose-invert max-w-none prose-p:my-2 prose-strong:text-cyan-200" dangerouslySetInnerHTML={{ __html: `<p>${html}</p>` }} />;
}

function MiniChart({ tone }: { tone: GalleryItem['tone'] }) {
  const stroke = tone === 'green' ? '#34d399' : tone === 'red' ? '#f87171' : tone === 'amber' ? '#fbbf24' : '#00e5ff';
  return (
    <svg className="h-full w-full" viewBox="0 0 220 140" preserveAspectRatio="none">
      <rect width="220" height="140" fill="#0d1117" />
      <line x1="0" y1="35" x2="220" y2="35" stroke="#1c2533" />
      <line x1="0" y1="75" x2="220" y2="75" stroke="#1c2533" />
      <line x1="0" y1="115" x2="220" y2="115" stroke="#1c2533" />
      <path d="M0,108 L35,96 L70,102 L105,76 L140,66 L175,45 L220,34" fill="none" stroke={stroke} strokeWidth="3" strokeLinecap="round" />
      <path d="M0,108 L35,96 L70,102 L105,76 L140,66 L175,45 L220,34 L220,140 L0,140 Z" fill={stroke} opacity="0.12" />
      <circle cx="220" cy="34" r="5" fill={stroke} />
    </svg>
  );
}

function DefaultInsightCards() {
  const defaults: GalleryItem[] = [
    { title: 'Chưa có nguồn RAG. Hãy gửi một câu hỏi để cập nhật danh sách bài liên quan.', source: 'NewsAI', label: 'RAG', tone: 'cyan' },
    { title: 'Backend FastAPI sẽ trả answer, sources, images và related_articles qua /api/chat.', source: 'FastAPI', label: '/api/chat', tone: 'green' },
    { title: 'Bật Debug để xem query_plan, Chroma path và thông tin retrieval.', source: 'Debug', label: 'Trace', tone: 'amber' },
  ];
  return defaults.map((item) => <GalleryCard key={item.title} item={item} />);
}

type GalleryItem = {
  title: string;
  source?: string;
  publishedAt?: string;
  url?: string;
  imageUrl?: string;
  label: string;
  tone: 'cyan' | 'green' | 'red' | 'amber' | 'indigo';
};

function buildGalleryItems(response?: ChatResponse): GalleryItem[] {
  if (!response) return [];
  const imageByArticle = new Map<string, ImageContext>();
  const imageByCitation = new Map<number, ImageContext>();
  for (const image of response.images || []) {
    if (image.article_id && imageUrlOf(image)) imageByArticle.set(image.article_id, image);
    if (image.citation_id && imageUrlOf(image)) imageByCitation.set(image.citation_id, image);
  }
  return (response.sources || []).slice(0, 8).map((source, index) => {
    const image = imageByArticle.get(source.article_id) || imageByCitation.get(source.citation_id);
    return {
      title: source.title || `Nguồn [${source.citation_id}]`,
      source: source.source,
      publishedAt: source.published_at,
      url: source.url,
      imageUrl: image ? imageUrlOf(image) : '',
      label: source.primary_topic || source.topic || `Score ${formatScore(source.score)}`,
      tone: index % 3 === 0 ? 'cyan' : index % 3 === 1 ? 'green' : 'indigo',
    };
  });
}

function buildCurrentContext({
  previousQuestion,
  previousResponse,
  selectedSource,
  filters,
}: {
  previousQuestion: string;
  previousResponse?: ChatResponse;
  selectedSource: SourceContext | null;
  filters: ChatFilters;
}): CurrentContext | undefined {
  if (!previousResponse) return undefined;
  return {
    selected_article_id: selectedSource?.article_id,
    selected_url: selectedSource?.url,
    selected_source: selectedSource?.source,
    selected_citation_id: selectedSource?.citation_id,
    previous_question: previousQuestion,
    previous_answer: previousResponse.answer,
    previous_sources: previousResponse.sources ?? [],
    previous_related_articles: previousResponse.related_articles ?? [],
    previous_images: previousResponse.images ?? [],
    previous_query_plan: previousResponse.query_plan,
    active_filters: filters,
  };
}

function lastUserQuestion(messages: ChatMessage[]): string {
  return [...messages].reverse().find((message) => message.role === 'user')?.text || '';
}

function imageUrlOf(image: ImageContext | ChatResponse['images'][number]): string {
  return image.url || image.image_url || '';
}

function externalUrl(value?: string): string {
  const url = String(value || '').trim();
  return /^https?:\/\//i.test(url) ? url : '';
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
}

function formatDate(value?: string): string {
  return value ? value.substring(0, 10) : 'mới';
}

function formatScore(value?: number): string {
  return typeof value === 'number' ? value.toFixed(2) : 'RAG';
}

function toneClass(tone: GalleryItem['tone']): string {
  if (tone === 'green') return 'border-emerald-300/30 bg-emerald-300/10 text-emerald-300';
  if (tone === 'red') return 'border-red-300/30 bg-red-300/10 text-red-300';
  if (tone === 'amber') return 'border-amber-300/30 bg-amber-300/10 text-amber-300';
  if (tone === 'indigo') return 'border-indigo-300/30 bg-indigo-300/10 text-indigo-300';
  return 'border-cyan-300/30 bg-cyan-300/10 text-cyan-300';
}
