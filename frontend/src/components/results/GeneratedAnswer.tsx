interface Props {
  answer: string;
}

export default function GeneratedAnswer({ answer }: Props) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-2">Generated Answer</h3>
      <div className="rounded border border-gray-200 bg-white p-4">
        <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">{answer}</p>
      </div>
    </div>
  );
}
