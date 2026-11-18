"""Append private, genuine reserved-evaluation evidence to the development report."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
import sys

ROOT = Path(__file__).resolve().parents[1]
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from pypdf import PdfReader, PdfWriter


def main():
    evidence = ROOT / 'results/evaluator'
    result = json.loads((evidence / 'final_evaluation.json').read_text())
    inspection = json.loads((evidence / 'inspection.json').read_text())
    ui = json.loads((evidence / 'ui_verification.json').read_text())
    out = ROOT / 'output/private'
    out.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 13
    story = []
    def text(value, style='BodyText'):
        story.extend([Paragraph(escape(str(value)), styles[style]), Spacer(1, 8)])
    def table(rows, widths):
        wrapped = [[Paragraph(escape(str(v)), styles['BodyText']) for v in row] for row in rows]
        t = Table(wrapped, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eef2f8')), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#dce3ed')), ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
        story.extend([t, Spacer(1, 12)])
    percent = lambda value: f'{value:.2%}' if value is not None else 'Unavailable'
    agg = result['aggregate']; freeze = result['freeze']
    text('Private final evaluator assessment', 'Title')
    text('LOCAL ONLY - RESERVED DATA / RESULTS - DO NOT PUBLISH', 'Heading2')
    text('This appendix records the final reserved evaluation separately from the preceding development experiments. No model was retrained, preprocessing refitted, thresholds changed or parameters selected using the reserved data. Reserved inputs, per-trip predictions, browser screenshots and this report remain in ignored local folders.')
    text('Freeze and integrity', 'Heading2')
    for key in ['created_at_utc', 'git_commit', 'model_sha256', 'pipeline_sha256', 'configuration_sha256']:
        text(f'{key}: {freeze[key]}')
    text('Feature/candidate/decoder and alignment configurations are recorded in results/evaluator/freeze.json. The selected artifact is artifacts/final/selected.joblib. The freeze was made before reserved-file inspection.')
    text('Trip-ID overlap with original partitions: ' + str(inspection.get('partition_overlap', 'See inspection evidence')))
    text(f"Discovered {inspection['trip_count']} actual trips. Valid inputs: {sum(r['input_status']=='valid' for r in inspection['trips'])}. References: {sum(r.get('reference_status')=='valid' for r in inspection['trips'])} valid, {sum(r.get('reference_status')=='missing' for r in inspection['trips'])} missing, {sum(r.get('reference_status')=='invalid' for r in inspection['trips'])} invalid. Originals unchanged: {result['originals_unchanged']}. Frozen pipeline unchanged: {result['pipeline_unchanged']}.")
    text(result['official_scoring_procedure'])
    text('Scoring: agreement with provisional derived labels', 'Heading2')
    text('The existing monotone ordered-reference alignment supplies provisional observation labels. Only aligned observations are eligible. This is not independent official observation-level ground truth. Arc-set F1 measures unordered overlap and is not an ordered-sequence score. No score was used to modify the frozen system.')
    table([['Method', 'Top-1 agreement', 'Direct continuity', 'Coverage'], ['Classifier', percent(agg['classifier']['top1_accuracy']), percent(agg['classifier']['direct_step_continuity']), percent(agg['classifier']['alignment_coverage'])], ['Classifier + HMM', percent(agg['hmm']['top1_accuracy']), percent(agg['hmm']['direct_step_continuity']), percent(agg['hmm']['alignment_coverage'])]], [125, 115, 115, 110])
    text(f"API: {agg['successful_trips']}/{agg['trip_count']} trips passed. Observations: {agg['total_observations']:,}. Total measured HTTP runtime: {agg['http_runtime_s']:.3f}s. Browser selection/display/export passed: {sum(r['passed'] for r in ui['trips'])}/{ui['trip_count']}. JavaScript errors: {len(ui['javascript_errors'])}. External requests: {len(ui['blocked_external_requests'])}.")
    story.append(PageBreak())
    text('Per-trip final evaluation', 'Heading1')
    rows = [['Trip', 'Points', 'Classifier', 'HMM', 'HTTP s', 'Checks']]
    for row in result['trips']:
        metrics = row.get('metrics') or {}
        rows.append([row['trip_id'], row.get('observations', '-'), percent(metrics.get('classifier', {}).get('top1_accuracy')), percent(metrics.get('hmm', {}).get('top1_accuracy')), f"{row['http_runtime_s']:.3f}", 'PASS' if row['checks_passed'] else 'FAIL'])
    table(rows, [76, 55, 95, 95, 65, 79])
    text('Checks cover HTTP status, original observation order/count, predicted arc bounds, expanded route connectivity, gap/reset handling, frozen decoder settings and recursive reference-field leakage. Repeated browser predictions and exported predictions were compared with the API evidence for deterministic equality.')
    for row in result['trips']:
        if row.get('error') or row.get('evaluation_warning'):
            text(f"Trip {row['trip_id']}: {row.get('error') or row.get('evaluation_warning')}")
    text('Launch the final demo', 'Heading2')
    relative = str(Path(inspection['trip_root']).relative_to(ROOT))
    mac = f'bash start.sh --evaluator-root "{relative}" --api-only'
    windows_relative = relative.replace('/', '\\')
    windows = f'start.bat --evaluator-root "{windows_relative}" --api-only'
    text('From the repository root on Laptop 1, macOS: ' + mac)
    text('Windows CMD: ' + windows)
    text('On Laptop 2: bash start.sh --ui-only (macOS), or start.bat --ui-only (Windows). Open http://localhost:8001 and set Connection settings to http://LAPTOP1_LAN_IP:8000. A local ignored .pathsnap-local.json selects the reserved root for normal starts on the evaluation machine.')
    text('Remaining limitations', 'Heading2')
    text('Reserved trips use the offline network map so their locations are not requested from external tile servers. A second physical laptop/LAN and receiving-terminal screenshot must still be demonstrated at the event. Missing/invalid references retain predictions and suppress accuracy. Development and reserved scores have separate evidence files. Team review, submission and disclosure remain the team\'s responsibility.')
    successful = next((row for row in ui['trips'] if row['passed']), None)
    if successful:
        story.append(PageBreak())
        text('Verified reserved evaluator interface (private)', 'Heading1')
        screenshot = evidence / 'ui' / (successful['trip_id'] + '.png')
        from PIL import Image as PILImage
        with PILImage.open(screenshot) as picture:
            width, height = picture.size
        display_width = min(500, 630 * width / height)
        story.append(Image(str(screenshot), width=display_width, height=display_width * height / width))
        text('Actual local UI-triggered inference. Reference labels and reference geometry are absent. Offline network mode prevents external tile requests. All discovered trips were tested; this image is one representative capture.')
    appendix = out / 'evaluator_appendix.pdf'
    SimpleDocTemplate(str(appendix), pagesize=A4, rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45).build(story)
    writer = PdfWriter()
    writer.append(PdfReader(ROOT / 'output/pdf/PathSnap_Technical_Report.pdf'))
    writer.append(PdfReader(appendix))
    final = out / 'PathSnap_Final_Evaluation.pdf'
    with final.open('wb') as stream:
        writer.write(stream)
    (evidence / 'LAUNCH.md').write_text('# Private evaluator demo\n\nmacOS, Laptop 1:\n\n```sh\n' + mac + '\n```\n\nWindows CMD, Laptop 1:\n\n```bat\n' + windows + '\n```\n\nLaptop 2: `bash start.sh --ui-only` or `start.bat --ui-only`. Set the API address to Laptop 1\'s reachable LAN address on port 8000.\n\nDo not publish this directory or the reserved input folder. Physical two-device connectivity remains an event check.\n')
    print(final)


if __name__ == '__main__':
    main()
