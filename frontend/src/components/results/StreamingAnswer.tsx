interface Props {
  tokens: string[];
  isStreaming: boolean;
  error?: string;
}

export default function StreamingAnswer({ tokens, isStreaming, error }: Props) {
  const text = tokens.join("");

  if (!text && !isStreaming && !error) return null;

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-2">LLM Answer</h3>
      <div className="rounded border border-gray-200 bg-white p-4">
        {(text || isStreaming) && (
          <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
            {text}
            {isStreaming && (
              <span className="inline-block w-2 h-4 ml-0.5 bg-blue-500 animate-pulse align-text-bottom" />
            )}
          </p>
        )}
        {error && (
          <p className="text-sm text-red-600 mt-2">{error}</p>
        )}
      </div>
    </div>
  );
}
