"""Record a pre-inference desk judgment for each of the 24×5 reviewed slots."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
# Positions are fixed and validated against the replayed retrieval output.
# Two applicable passages can independently answer a question.
ROWS = {
    'T01': ('vna_pet_booking caav_liquid2016 pq_port.29 air_ped.05 foreign_cash_border', '00000'),
    'T02': ('rail_discount_overlap rail_discount_senior air_senior_fare air_disability_fare e_id.02', '21100'),
    'T03': ('rail_discount_senior air_senior_fare pq_port.37 air_disability_fare vin_book.09', '21000'),
    'T04': ('pas_lost.00 pas_lost.01 pas_lost.10 pas_lost.06 pas_lost.11', '00000'),
    'T05': ('air_comp.09 park_pack.12 visa_types.07 air_family.10 air_infant_birth', '00002'),
    'T06': ('air_delay_refund air_comp.16 air_comp.00 air_comp.06 air_comp.09', '20000'),
    'T07': ('air_weather air_comp.17 air_delay_refund air_comp.09 air_comp.15', '21000'),
    'T08': ('vat_departure foreign_cash_border vn_departure_old visa_jp.09 border_permit_loss_abroad', '20000'),
    'T09': ('air_family.10 passport_first_office park_pack.19 air_family.08 park_pack.18', '02000'),
    'T10': ('card_passport_expiry visa_de_15 visa_substitute bca_old_format vn_departure_old', '20000'),
    'T11': ('vn_visa_reentry air_in.00 vn_departure_old foreign_cash_border visa_substitute', '10002'),
    'T12': ('visa_de_15 vn_departure_old vn_visa_reentry evisa_entry_types vna_residency', '20100'),
    'T13': ('evisa_limit_90 evisa_entry_types vn_departure_old air_comp.06 air_comp.10', '22000'),
    'T14': ('evisa_entry_types vn_visa_reentry vn_departure_old evisa_allnationalities air_id.02', '20010'),
    'T15': ('border_vn_cn vn_departure_old air_cover.00 evisa_entry_types air_in.00', '00000'),
    'T16': ('air_ped.09 border_permit_loss_abroad passport_report air_comp.06 air_comp.17', '02000'),
    'T17': ('vn_departure_old metro_launch.03 visa_de_15 air_family.07 evisa_entry_types', '20000'),
    'T18': ('foreign_cash_border vn_departure_old border_permit_loss_abroad vat_departure pq_port.29', '20000'),
    'T19': ('passport_child_duration vn_visa_reentry vn_departure_old passport_restore air_infant_birth', '20000'),
    'T20': ('passport_restore passport_report vn_visa_reentry bca_old_format pas_lost.10', '21000'),
    'T21': ('pq_port.37 air_family.00 vin_book.18 air_comp.06 air_disability_fare', '00000'),
    'T22': ('air_disability_fare pq_port.17 metro_fare.01 air_senior_fare pq_port.16', '20000'),
    'T23': ('air_certified_copy air_id.01 air_id.10 air_infant_birth air_id.09', '21110'),
    'T24': ('air_senior_fare rail_discount_senior vin_book.18 metro_fare.01 pq_port.16', '22000'),
}
NOTES = {
    'T01':'Airline pet booking is a mode mismatch despite matching dogs and cats.',
    'T02':'One statutory fare amount does not resolve stacking.',
    'T03':'The standalone NĐ06 card supplies the rate but omits the eligibility age.',
    'T04':'Five consular passages concern abbreviated issuance abroad.',
    'T05':'The domestic infant rule accepts a birth notification at three weeks. International chaperone rules do not apply.',
    'T06':'Overbooking articles describe a different causal mechanism. Grade 2 resolves the two- and five-hour options and the trigger for advance compensation, without offering distance-dependent rates.',
    'T07':'A mention of the compensation instrument is incomplete without the weather exemption.',
    'T08':'Both rate and pre-boarding refund timing are required.',
    'T09':'Judge only domestic procedure. Family travel text describes different controls.',
    'T10':'The card must end at least 30 days before passport expiry. Entry-document substitutions do not resolve that condition.',
    'T11':'General entry conditions mention exceptions but do not name the two substituted cards.',
    'T12':'General exemption conditions do not name Germany or the then-applicable maximum of 15 days.',
    'T13':'Both the amended act and the later official implementation report establish the 15 August rule.',
    'T14':'Nationwide eligibility provides e-visa background but does not answer number of entries.',
    'T15':'Eligibility to obtain a border pass does not establish how long it stays valid.',
    'T16':'Domestic or international loss of an ordinary passport is a different document from a border pass.',
    'T17':'Foreign-national entry conditions, metro travel and child air travel do not give general citizen exit requirements.',
    'T18':'Neither a general passport exit rule nor tourist VAT refunds settle the cash declaration and paperwork.',
    'T19':'The adult passport rule cannot be applied retroactively to issuance at age 13.',
    'T20':'The report-of-loss provision traces cancellation, but does not establish eligibility for restoration.',
    'T21':'An operator page contents list does not state the statutory fare cutoff or relevant age rule.',
    'T22':'Ferry and metro concessions and senior air fare rules do not establish the domestic air disability concession.',
    'T23':'The carrier copy text is a later snapshot without historical validity metadata. The archived infant document provision omits the certified-copy rule.',
    'T24':'Both versions of the evidence card reproduce the qualifying passenger-air provision.',
}
questions = json.loads((ROOT/'reader_worksheet.json').read_text())
assert {q['id'] for q in questions} == set(ROWS) == set(NOTES)
labels = {}
for q in questions:
    ids, grades = ROWS[q['id']]
    checked = ids.split()
    assert checked == q['native'], (q['id'], checked, q['native'])
    labels[q['id']] = dict(zip(checked, map(int, grades), strict=True))
(ROOT/'reader_judgments.json').write_text(json.dumps(labels, ensure_ascii=False, indent=2)+'\n')
(ROOT/'reader_judgment_notes.json').write_text(json.dumps(NOTES, ensure_ascii=False, indent=2)+'\n')
