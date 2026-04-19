from proposal_generator import run_rag1
from scientific_advisor import run_rag2


MAX_ROUNDS = 5

def should_stop(advice: dict, round_idx: int) -> bool:
    status = advice.get("bom_check", {}).get("status", "").strip()

    if advice.get("parse_failed", False) and round_idx >= 1:
        return True

    if status == "strongly_feasible" and round_idx >= 1:
        return True

    if status == "mostly_feasible" and round_idx >= 2:
        return True

    return False

def print_round_summary(round_idx: int, rag1_output: dict, rag2_advice: dict) -> None:
    proposal = rag1_output.get("rag1_proposal", "")
    status = rag2_advice.get("bom_check", {}).get("status", "unknown")
    reason = rag2_advice.get("bom_check", {}).get("reason", "")

    print("\n" + "=" * 100)
    print(f"ROUND {round_idx + 1} SUMMARY")
    print("=" * 100)
    print(f"RAG2 status: {status}")
    print(f"Reason: {reason}")

    msg = rag2_advice.get("message_to_rag1", "")
    if msg:
        print(f"\nMessage to RAG1:\n{msg}")

    narrowing = rag2_advice.get("narrowing_advice", [])
    if narrowing:
        print("\nTop narrowing advice:")
        for i, item in enumerate(narrowing[:3], 1):
            advice_text = item.get("advice", "")
            why_text = item.get("why", "")
            print(f"{i}. {advice_text}")
            if why_text:
                print(f"   why: {why_text}")

    if proposal:
        print("\nLatest proposal preview:")
        print(proposal[:1500])


def main():
    rag2_feedback = None
    rag1_output = None
    rag2_advice = None
    cached_exp_evidence = None
    cached_sci_evidence = None

    for round_idx in range(MAX_ROUNDS):
        print("\n" + "=" * 100)
        print(f"MULTI-AGENT ROUND {round_idx + 1}")
        print("=" * 100)

        # 1) RAG1 generates or revises the experiment proposal
        rag1_output = run_rag1(
            rag2_feedback=rag2_feedback,
            cached_exp_evidence=cached_exp_evidence,
            save_output=True,
        )

        if cached_exp_evidence is None:
            cached_exp_evidence = rag1_output.get("cached_exp_evidence")

        # 2) RAG2 critiques the current proposal
        rag2_result = run_rag2(
            rag1_output=rag1_output,
            cached_sci_evidence=cached_sci_evidence,
            save_output=True,
        )

        rag2_advice = rag2_result["rag2_advice"]

        if cached_sci_evidence is None:
            cached_sci_evidence = rag2_result["cached_sci_evidence"]

        # 3) Print round summary
        print_round_summary(round_idx, rag1_output, rag2_advice)

        # 4) Decide whether to stop
        if should_stop(rag2_advice, round_idx):
            print("\nStopping condition met.")
            break

        # 5) Pass advice directly to next RAG1 round
        rag2_feedback = rag2_advice

    print("\n" + "=" * 100)
    print("MULTI-AGENT LOOP FINISHED")
    print("=" * 100)

    return {
        "final_rag1_output": rag1_output,
        "final_rag2_advice": rag2_advice,
    }


if __name__ == "__main__":
    main()
