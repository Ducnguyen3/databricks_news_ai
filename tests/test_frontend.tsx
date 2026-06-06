import React, { useState } from 'react';

// ==========================================
// 1. TYPES DEFINITIONS (Tích hợp trực tiếp)
// ==========================================
export type SourceName = 'vnexpress' | 'cafef' | 'genk' | 'diendandoanhnghiep';
export type Topic = 'technology_ai_internet' | 'economy_finance_stock' | 'politics_society' | 'world_geopolitics' | 'business_startup' | 'real_estate' | 'lifestyle_education_health_entertainment' | 'general_news';
export type Domain = 'cong_nghe' | 'tai_chinh' | 'bat_dong_san' | 'doi_song' | 'chinh_tri_xa_hoi' | 'the_gioi' | 'startup' | 'all';

export interface ChatFilters {
  time_range_days: number;
  sources: SourceName[];
  topic: Topic | '';
  domain: Domain | '';
  ticker: string;
  top_k: number;
}

export interface ChatResponse {
  answer: string;
  intent?: string;
  topic?: string;
  domain?: string;
  ticker?: string;
  query_plan?: {
    intent?: string;
    primary_topic?: string;
    domain?: string;
    ticker?: string;
    entities?: string[];
    time_range_days?: number;
    sources?: string[];
    top_k?: number;
    need_images?: boolean;
  };
  sources: Array<{
    citation_id: number;
    article_id: string;
    title: string;
    source: string;
    url: string;
    published_at?: string;
    score?: number;
    snippet?: string;
  }>;
  images: Array<{
    url: string;
    caption?: string;
    alt?: string;
    citation_id?: number;
  }>;
  related_articles?: Array<{ title: string; source: string; url: string; }>;
  debug?: {
    retrieved_chunks?: any[];
    prompt_preview?: string;
  };
}

// ==========================================
// 2. MOCK DATA FOR 3 TEST QUESTIONS
// ==========================================
const MOCK_ANSWERS: Record<string, ChatResponse> = {
  "ai": {
    answer: "Trong 14 ngày qua, làn sóng AI tại Việt Nam ghi nhận nhiều chuyển biến mạnh mẽ [1]. Các doanh nghiệp công nghệ lớn đang đẩy mạnh việc tích hợp các mô hình ngôn ngữ lớn (LLM) vào hệ thống vận hành nội bộ nhằm tối ưu chi phí [2]. Nổi bật là xu hướng xây dựng hệ thống RAG (Retrieval-Augmented Generation) kết hợp dữ liệu doanh nghiệp riêng tư trên hạ tầng điện toán đám mây an toàn.",
    intent: "latest_news",
    topic: "technology_ai_internet",
    domain: "cong_nghe",
    ticker: "",
    query_plan: {
      intent: "latest_news",
      primary_topic: "technology_ai_internet",
      domain: "cong_nghe",
      time_range_days: 14,
      sources: ["genk", "vnexpress"],
      top_k: 10,
      need_images: false
    },
    sources: [
      { citation_id: 1, article_id: "art_001", title: "Thị trường AI Việt Nam tăng tốc nửa đầu năm 2026", source: "genk", url: "https://genk.vn", published_at: "2026-06-01", score: 0.92, snippet: "Nhiều doanh nghiệp Việt bắt đầu ứng dụng triệt để Generative AI vào quy trình phân tích dữ liệu tự động." },
      { citation_id: 2, article_id: "art_002", title: "Giải pháp RAG tối ưu hóa tri thức doanh nghiệp", source: "vnexpress", url: "https://vnexpress.net", published_at: "2026-05-28", score: 0.85, snippet: "Kiến trúc RAG giải quyết triệt để bài toán Hallucination (gọi là ảo tưởng dữ liệu) của các mô hình ngôn ngữ lớn." }
    ],
    images: [],
    related_articles: [
      { title: "Nâng cấp hạ tầng Databricks xử lý dữ liệu lớn", source: "vnexpress", url: "https://vnexpress.net" }
    ],
    debug: {
      retrieved_chunks: [
        { chunk_id: "c1", text: "Thị trường AI Việt Nam tăng tốc...", score: 0.92 },
        { chunk_id: "c2", text: "Kiến trúc RAG giải quyết triệt để...", score: 0.85 }
      ],
      prompt_preview: "SYSTEM: Bạn là trợ lý tin tức chuyên nghiệp...\nCONTEXT: [1] Thị trường AI Việt Nam... [2] Giải pháp RAG...\nUSER: Tin AI mới nhất trong 14 ngày?"
    }
  },
  "hpg": {
    answer: "Tập đoàn Hòa Phát (HPG) vừa công bố sản lượng sản xuất thép đạt mức kỷ lục mới trong tháng qua [1]. Doanh thu thuần và lợi nhuận sau thuế giữ vững đà tăng trưởng ổn định nhờ việc tối ưu hóa chi phí vận hành lò cao và sự hồi phục từ thị trường bất động sản dân dụng [2].",
    intent: "financial_query",
    topic: "economy_finance_stock",
    domain: "tai_chinh",
    ticker: "HPG",
    query_plan: {
      intent: "financial_query",
      primary_topic: "economy_finance_stock",
      domain: "tai_chinh",
      ticker: "HPG",
      time_range_days: 14,
      sources: ["cafef", "diendandoanhnghiep"],
      top_k: 10,
      need_images: true
    },
    sources: [
      { citation_id: 1, article_id: "art_003", title: "Hòa Phát công bố kết quả kinh doanh vượt kỳ vọng", source: "cafef", url: "https://cafef.vn", published_at: "2026-06-03", score: 0.95, snippet: "Sản lượng thép cuộn cán nóng (HRC) đóng góp tỷ trọng lớn vào doanh thu quý này của tập đoàn." },
      { citation_id: 2, article_id: "art_004", title: "Thị trường thép xây dựng đón tín hiệu tích cực", source: "diendandoanhnghiep", url: "https://diendandoanhnghiep.vn", published_at: "2026-06-02", score: 0.88, snippet: "Nhu cầu tiêu thụ nội địa tăng trưởng trở lại là bệ phóng vững chắc cho các doanh nghiệp đầu ngành." }
    ],
    images: [
      { url: "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600", caption: "Nhà kho tổ hợp luyện kim thép Hòa Phát vận hành hết công suất", alt: "Nha may hoa phat", citation_id: 1 }
    ],
    related_articles: [
      { title: "Diễn biến giá quặng sắt thế giới tuần này", source: "cafef", url: "https://cafef.vn" }
    ],
    debug: {
      retrieved_chunks: [{ chunk_id: "c3", text: "Hòa Phát công bố kết quả kinh doanh..." }],
      prompt_preview: "SYSTEM: Bạn là chuyên gia phân tích tài chính...\nCONTEXT: [1] Hòa Phát công bố...\nUSER: HPG có gì mới?"
    }
  },
  "ukraine": {
    answer: "Hệ thống ghi nhận một số bài viết tiêu điểm liên quan đến diễn biến địa chính trị khu vực Ukraine và các tác động kinh tế đi kèm [1]. Các luồng tin tức tập trung vào hoạt động chuỗi cung ứng và vận tải logistics xuyên biên giới.",
    intent: "world_news",
    topic: "world_geopolitics",
    domain: "the_gioi",
    ticker: "",
    query_plan: { intent: "world_news", primary_topic: "world_geopolitics", domain: "the_gioi", time_range_days: 14, sources: [], top_k: 5, need_images: true },
    sources: [
      { citation_id: 1, article_id: "art_005", title: "Tác động địa chính trị toàn cầu đến chuỗi cung ứng 2026", source: "vnexpress", url: "https://vnexpress.net", published_at: "2026-06-04", score: 0.89, snippet: "Hậu cần logistics đường biển chịu ảnh hưởng kéo dài từ các xung đột chưa có hồi kết." }
    ],
    images: [
      { url: "https://broken-link-image-test.com/invalid.jpg", caption: "Ảnh lỗi link test - Sẽ tự động ẩn đi không làm lỗi layout", citation_id: 1 },
      { url: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=600", caption: "Bản đồ phân tích tác động chuỗi cung ứng vận tải hàng hải quốc tế", citation_id: 1 }
    ],
    sources_count: 1
  }
};

// ==========================================
// 3. MAIN COMPONENT
// ==========================================
export default function ChatPage() {
  const [question, setQuestion] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [showConfig, setShowConfig] = useState<boolean>(false);
  const [debugMode, setDebugMode] = useState<boolean>(false);
  const [hiddenImages, setHiddenImages] = useState<Record<number, boolean>>({});

  const [filters, setFilters] = useState<ChatFilters>({
    time_range_days: 14,
    sources: [],
    topic: '',
    domain: '',
    ticker: '',
    top_k: 10,
  });

  const handleAsk = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResponse(null);
    setHiddenImages({});

    // Chuẩn hóa chuỗi tìm mock data
    const normalizedQuery = question.toLowerCase();
    
    setTimeout(() => {
      let matchedKey = "";
      if (normalizedQuery.includes("ai")) matchedKey = "ai";
      else if (normalizedQuery.includes("hpg") || normalizedQuery.includes("hòa phát")) matchedKey = "hpg";
      else if (normalizedQuery.includes("ukraine")) matchedKey = "ukraine";

      if (matchedKey && MOCK_ANSWERS[matchedKey]) {
        setResponse(MOCK_ANSWERS[matchedKey]);
      } else {
        // Fallback Response HTTP 200 rỗng nguồn theo đặc tả yêu cầu
        setResponse({
          answer: "Dựa trên các bài báo hệ thống đã thu thập hiện tại, tôi chưa có thông tin đủ để trả lời câu hỏi này.",
          sources: [],
          images: []
        });
      }
      setLoading(false);
    }, 800); // Tạo hiệu ứng loading mượt mà
  };

  const handleClear = () => {
    setQuestion('');
    setResponse(null);
    setError(null);
    setHiddenImages({});
  };

  const toggleSource = (src: SourceName) => {
    setFilters(prev => ({
      ...prev,
      sources: prev.sources.includes(src) 
        ? prev.sources.filter(s => s !== src) 
        : [...prev.sources, src]
    }));
  };

  const selectSuggested = (txt: string, updatedFilters: Partial<ChatFilters>) => {
    setQuestion(txt);
    setFilters(prev => ({ ...prev, ...updatedFilters }));
  };

  const handleImageError = (index: number) => {
    setHiddenImages(prev => ({ ...prev, [index]: true }));
  };

  return (
    <div class="h-screen w-screen flex flex-col bg-slate-50 text-slate-800 font-sans overflow-hidden">
      
      {/* HEADER BAR */}
      <header class="bg-white border-b border-slate-200 px-6 py-4 flex justify-between items-center z-10 shrink-0">
        <div class="flex items-center space-x-3">
          <div class="bg-slate-900 text-white p-2 rounded-lg font-black text-sm tracking-wider">RAG</div>
          <div>
            <h1 class="text-base font-bold text-slate-900 tracking-tight">News RAG Assistant</h1>
            <p class="text-xs text-slate-500">Hybrid Search + RAG độc lập dựa trên Databricks Gold Table</p>
          </div>
        </div>
        <div class="flex items-center space-x-3">
          <label class="flex items-center space-x-2 text-xs font-semibold text-slate-600 cursor-pointer bg-slate-100 px-3 py-1.5 rounded-lg hover:bg-slate-200 transition">
            <input 
              type="checkbox" 
              checked={debugMode} 
              onChange={(e) => setDebugMode(e.target.checked)}
              class="rounded text-slate-900 focus:ring-slate-900 border-slate-300 w-4 h-4"
            />
            <span>Bật chế độ Debug</span>
          </label>
        </div>
      </header>

      {/* WORKSPACE LAYOUT */}
      <div class="flex flex-1 overflow-hidden w-full">
        
        {/* MAIN CONTAINER (LEFT & CENTER COLUMN) */}
        <div class="flex-1 flex flex-col justify-between overflow-y-auto p-4 md:p-6 bg-slate-50 border-r border-slate-200">
          <div class="max-w-3xl mx-auto w-full space-y-6">
            
            {/* COLLAPSIBLE FILTER PANEL */}
            <div class="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden transition-all duration-200">
              <button 
                type="button"
                onClick={() => setShowConfig(!showConfig)}
                class="flex items-center justify-between w-full text-xs font-bold text-slate-700 p-4 hover:bg-slate-50 transition"
              >
                <div class="flex items-center space-x-2">
                  {/* SVG Gear Icon */}
                  <svg class="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                  <span>Cấu hình bộ lọc nâng cao</span>
                  <span class="text-[11px] font-normal text-slate-400">({filters.time_range_days === 0 ? 'Tất cả' : `${filters.time_range_days} ngày`}, {filters.sources.length || 'Tất cả'} nguồn)</span>
                </div>
                {/* SVG Chevron Arrow */}
                <svg class={`w-4 h-4 text-slate-400 transition-transform ${showConfig ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
              </button>

              {showConfig && (
                <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 p-4 border-t border-slate-100 bg-slate-50/50 text-xs">
                  <div>
                    <label class="block font-bold text-slate-600 mb-1">Thời gian quét tin</label>
                    <select 
                      value={filters.time_range_days}
                      onChange={(e) => setFilters({...filters, time_range_days: Number(e.target.value)})}
                      class="w-full bg-white border border-slate-200 rounded-lg px-2 py-2 outline-none focus:border-slate-400"
                    >
                      <option value={7}>7 ngày qua</option>
                      <option value={14}>14 ngày qua</option>
                      <option value={30}>30 ngày qua</option>
                      <option value={0}>Tất cả lịch sử</option>
                    </select>
                  </div>

                  <div>
                    <label class="block font-bold text-slate-600 mb-1">Chủ đề phân tách</label>
                    <select 
                      value={filters.topic}
                      onChange={(e) => setFilters({...filters, topic: e.target.value as Topic})}
                      class="w-full bg-white border border-slate-200 rounded-lg px-2 py-2 outline-none focus:border-slate-400"
                    >
                      <option value="">Tất cả chủ đề</option>
                      <option value="technology_ai_internet">Công nghệ - AI - Internet</option>
                      <option value="economy_finance_stock">Kinh tế - Tài chính - Chứng khoán</option>
                      <option value="world_geopolitics">Quốc tế - Địa chính trị</option>
                    </select>
                  </div>

                  <div>
                    <label class="block font-bold text-slate-600 mb-1">Mã cổ phiếu / Ticker</label>
                    <input 
                      type="text"
                      placeholder="Ví dụ: HPG, FPT..."
                      value={filters.ticker}
                      onChange={(e) => setFilters({...filters, ticker: e.target.value.toUpperCase()})}
                      class="w-full bg-white border border-slate-200 rounded-lg px-2 py-2 outline-none focus:border-slate-400 placeholder:text-slate-300"
                    />
                  </div>

                  <div class="sm:col-span-2 md:col-span-3">
                    <label class="block font-bold text-slate-600 mb-1.5">Giới hạn nguồn báo chí</label>
                    <div class="flex flex-wrap gap-2">
                      {(['vnexpress', 'cafef', 'genk', 'diendandoanhnghiep'] as SourceName[]).map((src) => (
                        <button
                          type="button"
                          key={src}
                          onClick={() => toggleSource(src)}
                          class={`px-3 py-1.5 rounded-lg border text-[11px] font-semibold uppercase tracking-wider transition ${
                            filters.sources.includes(src)
                              ? 'bg-slate-900 border-slate-900 text-white'
                              : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-100'
                          }`}
                        >
                          {src === 'diendandoanhnghiep' ? 'Diễn đàn DN' : src}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* RESPONSE VIEWER AREA */}
            <div class="space-y-4">
              {loading && (
                <div class="p-6 bg-white border border-slate-200 rounded-xl shadow-sm flex flex-col items-center justify-center space-y-3">
                  <div class="w-5 h-5 border-2 border-slate-900 border-t-transparent rounded-full animate-spin"></div>
                  <div class="text-xs font-medium text-slate-500">Đang thực hiện Hybrid Search và tạo câu trả lời...</div>
                </div>
              )}

              {response ? (
                <div class="bg-white border border-slate-200 rounded-xl shadow-sm p-5 md:p-6 space-y-4 animate-fade-in">
                  {/* Badges thông tin định tuyến từ Router */}
                  {(response.intent || response.topic || response.ticker) && (
                    <div class="flex flex-wrap gap-1.5 text-[10px] font-bold uppercase tracking-wider">
                      {response.intent && <span class="bg-indigo-50 text-indigo-600 border border-indigo-100 px-2 py-0.5 rounded-md">Intent: {response.intent}</span>}
                      {response.topic && <span class="bg-emerald-50 text-emerald-600 border border-emerald-100 px-2 py-0.5 rounded-md">Topic: {response.topic}</span>}
                      {response.ticker && <span class="bg-amber-50 text-amber-700 border border-amber-200 px-2.5 py-0.5 rounded-md">Ticker: {response.ticker}</span>}
                    </div>
                  )}
                  {/* Khung nội dung văn bản chính */}
                  <div class="text-slate-800 text-sm leading-relaxed whitespace-pre-wrap selection:bg-orange-100">
                    {response.answer}
                  </div>
                </div>
              ) : (
                !loading && (
                  <div class="bg-white border border-slate-200 rounded-xl p-6 text-center shadow-sm max-w-md mx-auto mt-8">
                    <div class="w-10 h-10 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-3 text-slate-600">📰</div>
                    <h3 class="font-bold text-slate-800 text-sm mb-1">Hệ thống Tổng hợp Tin tức RAG</h3>
                    <p class="text-xs text-slate-400 mb-5">Hệ thống demo tự động crawl và phân tách dữ liệu trên Databricks. Hãy chọn thử các mẫu câu test nhanh:</p>
                    <div class="space-y-2 text-left text-xs">
                      <button onClick={() => selectSuggested("Tin AI mới nhất trong 14 ngày?", {topic: "technology_ai_internet", time_range_days: 14})} class="w-full p-2.5 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg text-slate-700 font-semibold transition text-left flex justify-between items-center">
                        <span>💡 Tin AI mới nhất trong 14 ngày?</span>
                        <span class="text-[10px] bg-slate-200 px-1.5 py-0.5 rounded text-slate-500">Test 1</span>
                      </button>
                      <button onClick={() => selectSuggested("HPG có gì mới?", {ticker: "HPG"})} class="w-full p-2.5 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg text-slate-700 font-semibold transition text-left flex justify-between items-center">
                        <span>📈 Mã cổ phiếu HPG có gì mới?</span>
                        <span class="text-[10px] bg-slate-200 px-1.5 py-0.5 rounded text-slate-500">Test 2</span>
                      </button>
                      <button onClick={() => selectSuggested("Có ảnh nào liên quan đến Ukraine không?", {time_range_days: 0, topic: "world_geopolitics"})} class="w-full p-2.5 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg text-slate-700 font-semibold transition text-left flex justify-between items-center">
                        <span>📸 Có ảnh nào liên quan đến Ukraine không?</span>
                        <span class="text-[10px] bg-slate-200 px-1.5 py-0.5 rounded text-slate-500">Test 3</span>
                      </button>
                    </div>
                  </div>
                )
              )}
            </div>

            {/* TECHNICAL DEBUG CODES PANEL */}
            {debugMode && response?.debug && (
              <div class="bg-slate-900 text-slate-200 rounded-xl p-4 text-xs font-mono space-y-3 max-h-[350px] overflow-y-auto shadow-inner border border-slate-800">
                <div class="flex items-center space-x-2 border-b border-slate-800 pb-2 text-orange-400 font-bold tracking-wide">
                  <span># VISUAL DEBUG ENGINE</span>
                </div>
                <div>
                  <div class="text-slate-500 font-bold mb-1">// Query Plan:</div>
                  <pre class="bg-slate-950 p-2 rounded overflow-x-auto text-emerald-400">{JSON.stringify(response.query_plan, null, 2)}</pre>
                </div>
                <div>
                  <div class="text-slate-500 font-bold mb-1">// Prompt Preview:</div>
                  <pre class="bg-slate-950 p-2 rounded overflow-x-auto whitespace-pre-wrap text-slate-300 leading-normal">{response.debug.prompt_preview || "N/A"}</pre>
                </div>
              </div>
            )}
          </div>

          {/* CHAT INPUT BOX FIXED BAR */}
          <div class="max-w-3xl mx-auto w-full pt-4 sticky bottom-0 bg-slate-50">
            <form onSubmit={handleAsk} class="bg-white border border-slate-200 rounded-xl p-1.5 shadow-md flex items-center space-x-2 focus-within:border-slate-400 transition-all">
              <input 
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Hỏi đáp thông tin thị trường, doanh nghiệp và báo chí đã crawl..."
                class="flex-1 bg-transparent border-0 outline-none text-sm px-3 text-slate-800 py-2"
                disabled={loading}
              />
              <div class="flex items-center space-x-1">
                {response && (
                  <button 
                    type="button" 
                    onClick={handleClear}
                    class="px-3 py-2 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-xs font-bold transition"
                  >
                    Reset
                  </button>
                )}
                <button 
                  type="submit"
                  disabled={loading || !question.trim()}
                  class="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-lg text-xs font-bold disabled:opacity-30 transition flex items-center space-x-1"
                >
                  <span>Gửi tin</span>
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* RIGHTS PANEL (SOURCES, IMAGES, RELATED ITEMS) */}
        <aside class="w-[360px] bg-white border-l border-slate-200 flex flex-col overflow-y-auto p-4 space-y-6 hidden lg:block shrink-0">
          
          {/* SEC 1: SOURCE CITATIONS */}
          <div>
            <h3 class="text-[10px] font-black text-slate-400 uppercase tracking-wider mb-3 flex items-center space-x-1.5">
              <span>📍 Nguồn trích dẫn từ RAG ({response?.sources?.length || 0})</span>
            </h3>
            {response && response.sources && response.sources.length > 0 ? (
              <div class="space-y-2.5">
                {response.sources.map((src) => (
                  <div key={src.citation_id} class="p-3 bg-slate-50 border border-slate-200 rounded-lg hover:border-slate-400 hover:bg-slate-50/20 transition text-xs flex flex-col space-y-1.5">
                    <div class="flex justify-between items-center">
                      <span class="inline-block bg-slate-900 text-white font-black px-1.5 py-0.5 rounded text-[9px]">
                        [{src.citation_id}]
                      </span>
                      <span class="text-[9px] font-extrabold uppercase text-slate-500 px-1.5 py-0.5 bg-slate-200 rounded tracking-wider">
                        {src.source}
                      </span>
                    </div>
                    <a href={src.url} target="_blank" rel="noreferrer" class="font-bold text-slate-900 hover:text-blue-600 transition line-clamp-2 leading-tight">
                      {src.title}
                    </a>
                    {src.published_at && <span class="text-[10px] text-slate-400">Ngày xuất bản: {src.published_at}</span>}
                    {src.snippet && <p class="text-slate-500 line-clamp-3 bg-white p-2 rounded border border-slate-100 text-[11px] leading-relaxed italic">"{src.snippet}"</p>}
                  </div>
                ))}
              </div>
            ) : (
              <p class="text-xs italic text-slate-400 bg-slate-50 p-3 rounded-lg border border-dashed border-slate-200 text-center">Chưa có nguồn được sử dụng cho phiên này.</p>
            )}
          </div>

          {/* SEC 2: IMAGES LIST */}
          <div>
            <h3 class="text-[10px] font-black text-slate-400 uppercase tracking-wider mb-3">📸 Hình ảnh liên quan ({response?.images?.filter((_, i) => !hiddenImages[i])?.length || 0})</h3>
            {response && response.images && response.images.length > 0 ? (
              <div class="space-y-3">
                {response.images.map((img, i) => {
                  // Tự động ẩn hoàn toàn khỏi cấu trúc cây DOM nếu lỗi link hoặc nằm trong danh sách ẩn
                  if (hiddenImages[i]) return null;
                  return (
                    <div key={i} class="border border-slate-200 rounded-lg overflow-hidden bg-slate-50 shadow-sm">
                      <img 
                        src={img.url} 
                        alt={img.alt || 'RAG News Thumbnail'} 
                        class="w-full h-36 object-cover"
                        onError={() => handleImageError(i)}
                      />
                      {img.caption && (
                        <div class="p-2 text-[11px] text-slate-500 border-t border-slate-100 bg-white leading-normal">
                          {img.citation_id && <span class="font-bold text-slate-800">[{img.citation_id}]</span>} {img.caption}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <p class="text-xs italic text-slate-400 bg-slate-50 p-3 rounded-lg border border-dashed border-slate-200 text-center">Không có hình ảnh đi kèm.</p>
            )}
          </div>

          {/* SEC 3: RELATED ARTICLES */}
          {response && response.related_articles && response.related_articles.length > 0 && (
            <div class="pt-2 border-t border-slate-100">
              <h3 class="text-[10px] font-black text-slate-400 uppercase tracking-wider mb-2.5">🔗 Bài viết liên quan khác</h3>
              <div class="space-y-1.5 text-xs">
                {response.related_articles.map((rel, idx) => (
                  <a key={idx} href={rel.url} target="_blank" rel="noreferrer" class="block p-2 bg-slate-50 hover:bg-indigo-50 hover:text-indigo-600 rounded border border-slate-200 text-slate-700 truncate font-semibold transition">
                    • [{rel.source.toUpperCase()}] {rel.title}
                  </a>
                ))}
              </div>
            </div>
          )}
        </aside>

      </div>
    </div>
  );
}