export type SourceName = 'vnexpress' | 'cafef' | 'genk' | 'diendandoanhnghiep';
export type Topic = 'tech_ai_internet' | 'technology_ai_internet' | 'economy_finance_stock' | 'politics_society' | 'world_geopolitics' | 'business_startup' | 'real_estate' | 'lifestyle_education_health_entertainment' | 'general_news';
export type Domain = 'cong_nghe' | 'tai_chinh' | 'bat_dong_san' | 'doi_song' | 'chinh_tri_xa_hoi' | 'the_gioi' | 'startup' | 'all';

export interface ChatFilters {
  time_range_days: number;
  sources: SourceName[];
  topic?: Topic | '';
  domain?: Domain | '';
  ticker?: string;
  top_k: number;
}

export interface SourceContext {
  citation_id?: number;
  article_id?: string;
  title?: string;
  source?: string;
  url?: string;
  published_at?: string;
  topic?: string;
  primary_topic?: string;
  snippet?: string;
  score?: number;
}

export interface ImageContext {
  url?: string;
  image_url?: string;
  caption?: string;
  credit?: string;
  source?: string;
  article_id?: string;
  article_title?: string;
  article_url?: string;
  citation_id?: number;
  is_representative?: boolean;
  type?: 'original' | 'generated' | string;
}

export interface CurrentContext {
  selected_article_id?: string;
  selected_url?: string;
  selected_source?: string;
  selected_citation_id?: number;
  previous_question?: string;
  previous_answer?: string;
  previous_sources?: SourceContext[];
  previous_related_articles?: SourceContext[];
  previous_images?: ImageContext[];
  previous_query_plan?: ChatResponse['query_plan'];
  active_filters?: ChatFilters;
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
    latest_article_count?: number;
    prefer_latest?: boolean;
    need_images?: boolean;
    answer_mode?: string;
    date_filter?: Record<string, unknown> | null;
    [key: string]: unknown;
  };
  sources: Array<{
    citation_id: number;
    id?: number;
    article_id: string;
    title: string;
    source: string;
    url: string;
    published_at?: string;
    primary_topic?: string;
    topic?: string;
    domain?: string;
    score?: number;
    snippet?: string;
  }>;
  images: Array<{
    url: string;
    image_url?: string;
    caption?: string;
    alt?: string;
    source?: string;
    article_id?: string;
    article_title?: string;
    article_url?: string;
    credit?: string;
    citation_id?: number;
    is_representative?: boolean;
    type?: 'original' | 'generated' | string;
  }>;
  generated_image_prompts?: Array<{
    article_id: string;
    article_title: string;
    source?: string;
    prompt: string;
    image_generation_prompt?: string;
  }>;
  related_articles?: Array<{ title: string; source: string; url: string; }>;
  debug?: {
    retrieved_chunks?: unknown[];
    prompt_preview?: string;
    chroma_path?: string;
    embedding_model?: string;
    collection?: string;
    [key: string]: unknown;
  };
}
