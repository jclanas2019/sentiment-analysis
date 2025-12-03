from jinja2 import Template

TEMPLATE = """
<html>
<head>
    <meta charset="utf-8" />
    <title>Reporte de Sentimientos</title>
    <style>
        body { font-family: Arial, sans-serif; padding:20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { padding: 8px 10px; border: 1px solid #ccc; }
        tr.correct { background:#e8ffe8; }
        tr.error   { background:#ffe8e8; }
    </style>
</head>
<body>
    <h2>Resultados Predicción Sentimental</h2>
    <table>
        <tr>
            <th>Texto</th>
            <th>Real</th>
            <th>Predicción</th>
            <th>Acierto</th>
        </tr>
        {% for r in rows %}
        <tr class="{{ 'correct' if r.is_correct else 'error' }}">
            <td>{{ r.text }}</td>
            <td>{{ r.true_sentiment }}</td>
            <td>{{ r.predicted_sentiment }}</td>
            <td>{{ '✔️' if r.is_correct else '❌' }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

def build_interactive_html(detailed_predictions):
    rows = detailed_predictions.to_dict(orient="records")
    html = Template(TEMPLATE).render(rows=rows)
    return html  


