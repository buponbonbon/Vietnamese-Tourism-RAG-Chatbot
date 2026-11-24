"""Compare chronological and reversed displays of the same independently verified GTFS events."""
from collections import Counter
from pathlib import Path
import json

from timed_analyze import parse_document,parse_question,flags
from timed_build import clock

ROOT=Path(__file__).resolve().parent


def display(doc,reverse):
    rows=list(zip(doc['names'],doc['sequence'],strict=True))
    if reverse:
        rows.reverse()
    head=f"{doc['label']} / {doc['route_name']} / {doc['date']}"
    stops=' | '.join(f'{name} ({clock(row[2])}, {clock(row[3])})' for name,row in rows)
    return head+'\nĐiểm dừng (giờ đến, giờ đi): '+stops


def main():
    test=json.loads((ROOT/'timed_eval.json').read_text())
    auditor=json.loads((ROOT/'timed_verification.json').read_text())
    assert len(test['candidates'])==auditor['reference_windows_checked']
    counts=Counter()
    decisions={}
    for q in test['questions']:
        ask=parse_question(q['text'])
        assert ask
        for pool in q['pools'].values():
            gold=set(pool['sufficient'])
            for pid in pool['candidates']:
                doc=test['candidates'][pid]
                decisions[q['id'],pid]=(ask,doc,pid in gold)
    for q,d,relevant in decisions.values():
        for flip in (False,True):
            p=parse_document(display(d,flip))
            assert len(p.mentions)==len(set(p.mentions)) and p.times
            mode='reverse_display' if flip else 'chronological_display'
            counts[mode,'pairs']+=1
            counts[mode,'gold']+=relevant
            for rule in ('mention_order','clock_order'):
                demoted,unknown=flags(q,p,rule)
                assert not unknown
                counts[mode,rule,'misclassified']+=demoted==relevant
                counts[mode,rule,'gold_demoted']+=demoted and relevant
    results={}
    for m in ('chronological_display','reverse_display'):
        results[m]={'pairs':counts[m,'pairs'],'gold':counts[m,'gold']}
        for rule in ('mention_order','clock_order'):
            results[m][rule]={'incorrect_compatibility':counts[m,rule,'misclassified'],
                              'sufficient_passages_demoted':counts[m,rule,'gold_demoted']}
        assert results[m]['clock_order']['incorrect_compatibility']==0
    (ROOT/'timed_display_invariance.json').write_text(json.dumps(results,indent=2)+'\n')
    print(results)


if __name__=='__main__':main()
