"""Build the review copy from the verified three-dataset refresh results."""

from pathlib import Path

import fitz
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from run_oracle_imitation_v29 import read, sha, write
from run_recovery_refresh_v36 import METHOD, OUT, v35
from summarize_refresh_v34_v35 import check_reference

OUTPUT = v35.v34.ROOT / 'output' / 'pdf'
SCENARIO = ['dataset', 'stride', 'offset']
NAMES = {'pamap2': 'PAMAP2', 'mhealth': 'MHEALTH', 'opportunity': 'OPPORTUNITY'}
ARMS = {'twoside_sel': 'V32', 'targeted_reserve_sel': 'V34 reserve', 'compact_reserve_sel': 'V35 reserve',
        'compact_sel': 'V35 standard', METHOD: 'V36'}


def build():
    verified = read(v35.OUT / 'comparison_verified.json')
    comparison = v35.OUT / 'matched_comparison.csv'
    assert verified['all_passed'] and verified['comparison_sha256'] == sha(comparison)
    assert read(v35.OUT / 'reserve_replay_audit.json')['passed']
    complete = read(OUT / 'summary_complete.json')
    assert complete['case_grids'] == 396 and complete['view_rows'] == 780
    results = pd.read_csv(comparison)
    prior = pd.read_csv(v35.bt.OUT / 'view_metrics.csv')
    check_reference(pd.read_csv(OUT / 'view_metrics.csv'), prior[prior.method == 'bt_supervised_cache_sel'])
    v36 = pd.read_csv(OUT / 'dataset_equal_metrics.csv').assign(notice_bytes=3)
    results = pd.concat([results, v36], ignore_index=True)
    results.to_csv(OUT / 'comparison_all_versions.csv', index=False)
    chosen = results[results.method.isin(ARMS)].set_index(SCENARIO + ['method'])
    published = results[results.method.str.startswith('bt_')]
    counts = results.groupby('method').all_checks_passed.sum()
    assert not published.all_checks_passed.any()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / 'refresh_comparison_v36.pdf'
    for name, filename in [('TNR', 'times.ttf'), ('TNRB', 'timesbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / filename)))
    body = ParagraphStyle('body', fontName='TNR', fontSize=11, leading=14, spaceAfter=8)
    title = ParagraphStyle('title', parent=body, fontName='TNRB', fontSize=20, leading=23, spaceAfter=12)
    head = ParagraphStyle('head', parent=body, fontName='TNRB', fontSize=12, spaceBefore=7)
    center = ParagraphStyle('center', parent=body, alignment=TA_CENTER, fontSize=9, leading=11)
    muted = colors.HexColor('#4d5661')
    note = ParagraphStyle('note', parent=body, textColor=muted, fontSize=9.5, leading=12)
    primary = v36[v36.stride == 1]
    story = [Paragraph('Sequential refresh on three wearable datasets', title),
             Paragraph(f'V36 passed {int(counts[METHOD])}/9 scenarios. On the primary grids it saved '
                       f'{primary.savings_fraction.min():.2%} to {primary.savings_fraction.max():.2%} of bytes '
                       'relative to complete push. Each version replays all 396 case-grids with frozen node models, '
                       'subject roles, activity labels and deployment gates.', body),
             Paragraph('Communication saving / reference disagreement (%)', head)]
    key_order = [(d, s, o) for d in NAMES for s, o in ((1, 0), (2, 0), (2, 1))]
    table_data = [['Dataset', 'Grid'] + [Paragraph(x, center) for x in ARMS.values()]]
    styles = [('FONTNAME', (0, 0), (-1, 0), 'TNRB'), ('FONTNAME', (0, 1), (-1, -1), 'TNR'),
              ('FONTSIZE', (0, 0), (-1, -1), 9), ('ALIGN', (2, 1), (-1, -1), 'CENTER'),
              ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 5),
              ('BOTTOMPADDING', (0, 0), (-1, -1), 5), ('LINEABOVE', (0, 0), (-1, 0), .8, colors.black),
              ('LINEBELOW', (0, 0), (-1, 0), .55, colors.black), ('LINEBELOW', (0, -1), (-1, -1), .8, colors.black)]
    for i, (dataset, stride, offset) in enumerate(key_order, 1):
        grid = 'Primary' if stride == 1 else f'Alt. {offset}'
        line = [NAMES[dataset] if stride == 1 else '', grid]
        if stride == 1 and i > 1:
            styles.append(('LINEABOVE', (0, i), (-1, i), .4, colors.HexColor('#adb5bd')))
        for column, method in enumerate(ARMS, 2):
            r = chosen.loc[(dataset, stride, offset, method)]
            mark = '' if r.all_checks_passed else '*'
            d = 100 * r.reference_disagreement
            line.append(f'{100*r.savings_fraction:.2f} / {d:.4f}{mark}')
            if r.all_checks_passed and method == METHOD:
                styles.append(('BACKGROUND', (column, i), (column, i), colors.HexColor('#e7f1ee')))
        table_data.append(line)
    table_data.append(['Passed', 'of 9'] + [str(int(counts[m])) for m in ARMS])
    styles += [('LINEABOVE', (0, -1), (-1, -1), .55, colors.black), ('FONTNAME', (0, -1), (-1, -1), 'TNRB')]
    story.extend([Table(table_data, colWidths=[81, 47, 70, 70, 70, 70, 70], style=TableStyle(styles)), Spacer(1, 8),
                  Paragraph('* One or more deployment gates failed. Alt. 0 and Alt. 1 are the two nonoverlap offsets.', note)])
    short = published[published.notice_bytes == 3]
    story.extend([Paragraph('Matched published comparison', head),
                  Paragraph('Four Beyond Thresholds adaptations select from their eleven confidence-bin budgets on B. '
                            f'With 3-byte notices and binding charges, they passed 0/9 scenarios and saved at most '
                            f'{short.savings_fraction.max():.2%}. All {verified["bt_fixed_scenario_points"]} fixed '
                            'adaptation points miss at least one agreement, F1, or model-call cap. '
                            'The adaptation uses a fixed primary and the common MiniRocket scores.', body),
                  Paragraph('Calibration and transport', head),
                  Paragraph('V34 charges 15 bytes per requested peer. V35 and V36 charge 3 bytes per peer '
                            'and 54 bytes per session to bind peer channels and share session ticks. '
                            'V36 selects the lowest-byte candidate that meets the B recovery gates, the F1 and call caps, '
                            'a 0.85% disagreement cap and positive savings for 75% of B subjects. '
                            'Its fallback uses the V35 reserve selection. The A gates are identical for every arm.', body),
                  Paragraph('All subjects come from the existing development study. Communication totals are packet replays '
                            'with a shared session clock in the compact setting. CoSS optimizes sensor and sampling-rate '
                            'choices, and Viveka evaluates data volume and energy under a different sensing protocol.', note),
                  Paragraph('Cooper and Marculescu, <i>Beyond Thresholds</i>, IoTDI 2024. '
                            '<link href="https://doi.org/10.1109/IoTDI61053.2024.00005">10.1109/IoTDI61053.2024.00005</link>. '
                            '<link href="https://www.dfki.de/en/web/research/projects-and-publications/publication/17091">CoSS</link>. '
                            '<link href="https://arxiv.org/abs/2608.05572">Viveka</link>.', note)])
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=48, rightMargin=48, topMargin=41, bottomMargin=40,
                      title='Sequential refresh comparison').build(story)
    with fitz.open(output) as document:
        assert len(document) == 1
        page = document[0]
        assert all(20 < block[0] < block[2] < page.rect.width-20 for block in page.get_text('blocks'))
        assert all(name in page.get_text() for name in NAMES.values())
        page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(OUT / 'comparison_preview.png')
    write(OUT / 'report_verification.json', dict(pages=1, table_scenarios=9, sha256=sha(output), passes=counts.to_dict()))
    print(output)


if __name__ == '__main__':
    build()
