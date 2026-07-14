/**
 * DataTable — Reusable enterprise table component
 * Supports alternating rows, bold headers, centered numeric columns
 */
export default function DataTable({ columns, rows }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((col, i) => (
              <th key={i} className={col.center ? 'center' : ''} style={col.width ? { width: col.width } : {}}>
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>
              {columns.map((col, ci) => {
                const val = row[col.key]
                return (
                  <td key={ci} className={col.center ? 'center' : ''}>
                    {col.render ? col.render(val, row) : val}
                  </td>
                )
              })}
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="center" style={{ color: 'var(--color-text-2)', padding: 24 }}>
                No data available
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
