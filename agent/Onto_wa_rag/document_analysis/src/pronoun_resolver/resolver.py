"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Optional

from .detector import MultilingualPronounDetector, Pronoun
from .llm import LLMProvider
from .models import ResolutionOutput, Replacement
from .spacy_backend import extract_context, ensure_model
from .utils import adjust_casing, APOS, safe_join_text

logger = logging.getLogger(__name__)


@dataclass
class ResolutionConfig:
    lang: str = "fr"
    batch_size: int = 8
    context_window: int = 500


class PronounResolver:
    def __init__(self, llm_provider: LLMProvider, config: ResolutionConfig | None = None):
        self.llm = llm_provider
        self.cfg = config or ResolutionConfig()
        self.detector = MultilingualPronounDetector(primary_language=self.cfg.lang)

        self.nlp = ensure_model(self.cfg.lang)
        if self.nlp is None:
            raise RuntimeError(f"Impossible de charger le modèle spaCy pour la langue '{self.cfg.lang}'.")

    def _group(self, pronouns: Sequence[Pronoun], max_items: int, max_span: int) -> List[List[Pronoun]]:
        if not pronouns:
            return []
        groups: List[List[Pronoun]] = []
        cur: List[Pronoun] = []
        cur_min = pronouns[0].position
        cur_max = pronouns[0].end_position
        for p in pronouns:
            new_min = min(cur_min, p.position) if cur else p.position
            new_max = max(cur_max, p.end_position) if cur else p.end_position
            span_ok = (new_max - new_min) <= max_span
            size_ok = (len(cur) + 1) <= max_items
            if cur and span_ok and size_ok:
                cur.append(p)
                cur_min, cur_max = new_min, new_max
            else:
                if cur:
                    groups.append(cur)
                cur = [p]
                cur_min, cur_max = p.position, p.end_position
        if cur:
            groups.append(cur)
        return groups

    def _build_prompt(self, full_text: str, group: Sequence[Pronoun], ctx: Dict[str, Any]) -> str:
        min_pos = max(0, min(p.position for p in group) - self.cfg.context_window // 2)
        max_pos = min(len(full_text), max(p.end_position for p in group) + self.cfg.context_window // 2)
        excerpt = full_text[min_pos:max_pos]

        pron_json = [
            {"pos": p.position, "text": p.text, "length": len(p.text)}  # pos cohérent avec Replacement
            for p in group
        ]

        return (
            "Tu es un expert en analyse grammaticale. Ta mission est de résoudre les coréférences de pronoms.\n"
            "Pour chaque pronom listé, tu dois fournir trois informations :\n"
            "1.  `text`: Le référent exact du pronom (ex: 'Jean').\n"
            "2.  `pronoun_type`: La fonction grammaticale du pronom. Choisis STRICTEMENT entre 'subject' (pour il, elle, ils, elles) ou 'indirect_object' (pour lui, leur).\n"
            "3.  `pos` et `length` : Recopie simplement la position et la longueur du pronom original qui t'ont été fournies.\n\n"
            "Exemple pour la phrase 'Elle lui parle.' où 'Elle' est 'Marie' et 'lui' est 'Jean':\n"
            " - Pour 'Elle': { 'pos': 0, 'length': 4, 'text': 'Marie', 'pronoun_type': 'subject' }\n"
            " - Pour 'lui': { 'pos': 5, 'length': 3, 'text': 'Jean', 'pronoun_type': 'indirect_object' }\n\n"
            f"--- CONTEXTE ---\n"
            f"Texte: \"{excerpt}\"\n"
            f"Entités candidates: {ctx.get('entities', [])}\n"
            f"--- FIN CONTEXTE ---\n\n"
            f"Pronoms à résoudre: {pron_json}\n\n"
            "Réponds UNIQUEMENT avec le modèle Pydantic fourni."
        )

    def _group_by_verb(self, doc, replacements: Sequence[Replacement]):
        groups = {}
        pos_to_replacement = {r.pos: r for r in replacements}
        for token in doc:
            if token.idx in pos_to_replacement:
                replacement = pos_to_replacement[token.idx]
                verb_head = token.head
                while verb_head.head != verb_head:
                    if verb_head.pos_ == "VERB":
                        break
                    verb_head = verb_head.head

                verb_head_index = verb_head.i
                if verb_head_index not in groups:
                    groups[verb_head_index] = {
                        "verb_token": verb_head,
                        "replacements": []
                    }
                groups[verb_head_index]["replacements"].append(replacement)
        return list(groups.values())

    def _needs_post_editing(self, sentence: str) -> bool:
        doc = self.nlp(sentence)

        prev_alpha = None
        for tok in doc:
            if tok.is_alpha:
                if prev_alpha is not None and tok.lower_ == prev_alpha.lower_:
                    return True
                prev_alpha = tok

        has_finite = any(
            (t.pos_ in ("VERB", "AUX")) and (t.morph.get("VerbForm") == ["Fin"])
            for t in doc
        )

        participles = [t for t in doc if "Part" in t.morph.get("VerbForm", [])]
        for p in participles:
            has_aux = any(
                c.dep_.startswith("aux") or c.pos_ == "AUX"
                for c in p.children
            ) or any(a.pos_ == "AUX" for a in p.ancestors)
            if not has_aux and not has_finite:
                return True

        if sentence.count("(") != sentence.count(")"):
            return True
        if sentence.count("[") != sentence.count("]"):
            return True
        if sentence.count("{") != sentence.count("}"):
            return True
        if sentence.count('"') % 2 == 1:
            return True
        if sentence.count("«") != sentence.count("»"):
            return True

        return False

    def _finalize_sentence(self, sentence: str, lang: str) -> str:
        prompt = (
            "Tu es un correcteur grammatical multilingue.\n"
            f"Réécris la phrase suivante en conservant STRICTEMENT le sens, la ponctuation logique "
            f"et les entités nommées. Utilise la langue détectée (hint: {lang}).\n"
            "Ne reformule pas inutilement ; corrige uniquement ce qui est grammaticalement nécessaire.\n"
            "IMPORTANT: Ne contracte pas 'à le' en 'au' ni 'à les' en 'aux' (conserve tel quel). Ne modifie pas l'ordre des segments déjà corrects.\n\n"
            f"Phrase:\n\"{sentence}\"\n\n"
            "Réponds UNIQUEMENT avec la phrase corrigée, sans guillemets."
        )
        try:
            out = self.llm.create_message([prompt], pydantic_model=None)
            if isinstance(out, str):
                return out.strip()
            if hasattr(out, "text"):
                return str(out.text).strip()
        except Exception as e:
            logger.warning(f"Post-editing LLM échoué : {e}")
        return sentence

    def _sentence_span_containing(self, doc, start_char: int, end_char: int):
        for sent in doc.sents:
            if sent.start_char <= start_char < sent.end_char or sent.start_char < end_char <= sent.end_char:
                return sent
        return list(doc.sents)[0] if list(doc.sents) else doc[:]

    # === Nouvelle application token-based, stable et locale ===
    def _apply(self, text: str, replacements: Sequence[Replacement]) -> str:
        if not replacements:
            return text

        doc = self.nlp(text)

        # Indexer les spans des remplacements sur les tokens spaCy et filtrer:
        # - garder uniquement les tokens POS=PRON
        # - garder uniquement pronoun_type in {'subject','indirect_object'}
        token_repls: List[Dict[str, Any]] = []
        for r in replacements:
            span = doc.char_span(r.pos, r.pos + r.length, alignment_mode="expand")
            if span is None:
                continue
            # En pratique, les pronoms devraient être des spans d'un token; on prend le token racine
            tok = span.root
            if tok.pos_ != "PRON":
                continue
            if getattr(r, "pronoun_type", None) not in ("subject", "indirect_object"):
                continue
            token_repls.append({"tok": tok, "rep": r})

        # Organiser par phrase
        by_sent: Dict[int, Dict[str, Any]] = {}
        for item in token_repls:
            tok = item["tok"]
            rep = item["rep"]
            sent = tok.sent
            sid = sent.start  # id stable de la phrase: index token de début
            data = by_sent.setdefault(sid, {
                "sent": sent,
                "subject_map": {},        # token_idx_in_sent -> new_text
                "iobj_ops": []            # list of dicts with keys: pron_tok, entity_text
            })
            if rep.pronoun_type == "subject":
                data["subject_map"][tok.i - sent.start] = rep.text
            else:
                data["iobj_ops"].append({
                    "pron_tok": tok,
                    "entity_text": rep.text
                })

        def find_main_verb(t):
            v = t.head
            while v.head != v:
                if v.pos_ == "VERB":
                    return v
                v = v.head
            return v if v.pos_ == "VERB" else t  # fallback

        def choose_insertion_index(sent, pron_tok):
            # Choisir où insérer "à <ENTITY>" dans la phrase sent
            # Heuristique:
            # 1) Si le verbe a un objet direct 'obj' dont le déterminant est possessif -> insérer avant cet OD
            # 2) Sinon -> insérer avant la ponctuation finale de la phrase
            verb = find_main_verb(pron_tok)
            obj_child = None
            for c in verb.children:
                if c.dep_ == "obj":
                    obj_child = c
                    break

            def has_possessive_det(np_head):
                if np_head is None:
                    return False
                for d in np_head.children:
                    if d.dep_ == "det":
                        # UD Possesif: Poss=Yes
                        if "Yes" in d.morph.get("Poss", []):
                            return True
                return False

            tokens = list(sent)
            if obj_child and has_possessive_det(obj_child):
                return max(0, obj_child.i - sent.start)  # avant OD
            # sinon, avant la ponctuation finale
            last = len(tokens)
            # trouver début de la traîne de ponctuation finale
            k = len(tokens) - 1
            while k >= 0 and tokens[k].is_punct:
                k -= 1
            return k + 1  # insertion avant les ponctuations finales

        # Reconstruire chaque phrase avec les modifs
        out_parts: List[str] = []
        prev_end = 0
        for sid, data in sorted(by_sent.items()):
            sent = data["sent"]
            subject_map = data["subject_map"]
            iobj_ops = data["iobj_ops"]

            # Ajoute le texte avant la phrase inchangé
            out_parts.append(text[prev_end:sent.start_char])

            tokens = list(sent)
            # Skip set pour pronoms iobj: on retire le clitique, on insérera plus tard "à NP"
            skip_idx = set()
            # insertions: map index token relatif -> liste de chaînes à insérer avant ce token
            insert_before: Dict[int, List[str]] = {}

            # Préparer insertions pour chaque iobjet
            for op in iobj_ops:
                pron_tok = op["pron_tok"]
                ent = op["entity_text"]
                ins_idx = choose_insertion_index(sent, pron_tok)
                insert_before.setdefault(ins_idx, []).append(f"à {ent}")
                # skipper le pronom iobj (et son whitespace)
                skip_idx.add(pron_tok.i - sent.start)

            # Recomposer la phrase
            buf: List[str] = []
            for j, tok in enumerate(tokens):
                # Insérer avant ce token s'il y a quelque chose à insérer
                if j in insert_before:
                    ins_list = insert_before[j]
                    # espace si nécessaire avant insertion
                    if buf and not buf[-1].endswith((" ", "\n")):
                        buf.append(" ")
                    buf.append(" ".join(ins_list))
                    # ajouter un espace si le token qui suit n'est pas une ponctuation
                    if not tok.is_punct:
                        buf.append(" ")

                # Skipper les pronoms iobj
                if j in skip_idx:
                    # on saute le token et son whitespace
                    continue

                # Remplacement sujet in-place
                if j in subject_map:
                    new_txt = adjust_casing(tok.text, subject_map[j])
                    buf.append(new_txt)
                    # conserver l'espace original du token
                    buf.append(tok.whitespace_)
                else:
                    buf.append(tok.text)
                    buf.append(tok.whitespace_)

            # Si on doit insérer en toute fin (après tous les tokens non-ponct), on a mappé sur l'index de la 1ère ponctuation
            # Rien de plus à faire ici car on l'a déjà inséré "avant" le token ponctuation concerné.

            phrase = "".join(buf)
            # Nettoyage léger des espaces multiples
            phrase = re.sub(r"\s{2,}", " ", phrase)
            # Eviter espace avant ponctuation
            phrase = re.sub(r"\s+([.,;:!?])", r"\1", phrase)

            # Post-edit si anomalie
            if self._needs_post_editing(phrase):
                phrase = self._finalize_sentence(phrase, self.cfg.lang)

            out_parts.append(phrase)
            prev_end = sent.end_char

        # Ajoute la fin du texte (phrases sans remplacements)
        out_parts.append(text[prev_end:])
        final_text = "".join(out_parts)
        return final_text

    def resolve(self, text: str) -> str:
        if not text or not text.strip():
            return text
        current = text
        # Utiliser le filtrage POS via spaCy
        pronouns = self.detector.find_pronouns(current, nlp=self.nlp)
        if not pronouns:
            return text

        groups = self._group(pronouns, self.cfg.batch_size, self.cfg.context_window)
        all_replacements: List[Replacement] = []

        for g in groups:
            ctx = extract_context(current, self.cfg.lang)
            prompt = self._build_prompt(current, g, ctx)
            out: ResolutionOutput = self.llm.create_message([prompt], pydantic_model=ResolutionOutput)
            if out and out.replacements:
                all_replacements.extend(out.replacements)

        if all_replacements:
            logger.info("Total replacements to apply: %d", len(all_replacements))
            current = self._apply(current, all_replacements)
            # Nettoyage final
            current = re.sub(r"\s{2,}", " ", current)
            current = re.sub(r"\s+([.,;:!?])", r"\1", current)

        return current

    def old_resolve(self, text: str) -> str:
        if not text or not text.strip():
            return text
        current = text
        pronouns = self.detector.find_pronouns(current, nlp=self.nlp)
        groups = self._group(pronouns, self.cfg.batch_size, self.cfg.context_window)
        total_applied = 0
        for g in groups:
            ctx = extract_context(current, self.cfg.lang)
            prompt = self._build_prompt(current, g, ctx)
            out: ResolutionOutput = self.llm.create_message([prompt], pydantic_model=ResolutionOutput)

            if out and out.replacements:
                applied_before = total_applied
                current = self._apply(current, out.replacements)
                total_applied += len(out.replacements)
                logger.info("Group resolved: %d replacements", total_applied - applied_before)

        logger.info("Total replacements applied: %d", total_applied)
        return current