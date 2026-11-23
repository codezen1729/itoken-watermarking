"""Deterministic open-text fallback using discourse boundaries and surface order."""
import re
import unicodedata
from functools import lru_cache

REASON=r'\b(because|since|given that|as long as|unless|when|whenever|while|after|before|if|provided that)\b'
EFFECT=r'\b(so that|in order to|which allows|which helps|which lets|thereby|so|thus)\b'
NEG=r"\b(must not|may not|cannot|can not|should not|do not|does not|never|forbid\w*|prohibit\w*|avoid\w*|withhold\w*|ban(?:s|ned)?)\b"
def clean(text):return ' '.join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?",text.lower())) or 'unspecified'

@lru_cache(maxsize=4096)
def fields(quote):
    norm=unicodedata.normalize('NFKC',quote).replace('’',"'").replace("n't"," not").strip()
    segments=[s.strip(' ,;:.') for s in re.split(r'(?<=[.!?])\s+',norm) if s.strip()]
    main=segments[0] if segments else norm
    reason=''; effect=' '.join(segments[1:])
    m=re.search(REASON,main,re.I)
    if m and m.start()==0 and ',' in main:reason,main=main.split(',',1)
    elif m and m.start()>0:reason=main[m.start():];main=main[:m.start()]
    m=re.search(EFFECT,main,re.I)
    if m and m.start()>0:effect=' '.join([main[m.start():],effect]);main=main[:m.start()]
    return clean(reason),clean(main),clean(effect),'avoid' if re.search(NEG,main,re.I) else 'do'

def canonical_step(step):
    q=step['quote'];r,a,e,s=fields(q)
    return dict(quote=q,premise=r,action=a,consequence=e,sign=s)
