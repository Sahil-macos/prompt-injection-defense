# ============================================================
# MODEL 1 — RISK FUSION
# MVNet Score + Policy Score
# ============================================================

MVNET_WEIGHT = 0.60
POLICY_WEIGHT = 0.40


# ============================================================
# RISK FUSION FUNCTION
# ============================================================

def calculate_risk(mvnet_score, policy_score, matched_rules=None):

    mvnet_score = float(mvnet_score)
    policy_score = float(policy_score)

    mvnet_score = max(0.0, min(1.0, mvnet_score))
    policy_score = max(0.0, min(1.0, policy_score))

    # --------------------------------------------------------
    # COMBINE MVNet + POLICY
    # --------------------------------------------------------

    risk_score = (
        MVNET_WEIGHT * mvnet_score
        + POLICY_WEIGHT * policy_score
    )

    # --------------------------------------------------------
    # STRONG POLICY OVERRIDE
    # --------------------------------------------------------

    # Explicit high-severity leakage/injection rules
    # should not be ignored even if MVNet confidence is low.

    strong_policy_rules = {
        "system_prompt_extraction",
        "hidden_instruction_extraction",
        "developer_message_extraction",
        "credential_extraction",
        "api_key_extraction",
        "secret_extraction",
        "data_exfiltration"
    }

    matched_rules = matched_rules or []

    has_strong_policy = any(
        rule in strong_policy_rules
        for rule in matched_rules
    )

    if has_strong_policy:
        risk_score = max(
            risk_score,
            0.90
        )

    risk_score = min(
        1.0,
        risk_score
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if risk_score > 0.30:
        decision = "SECONDARY_VERIFICATION"
    else:
     decision = "LOW_RISK"

    return {
        "mvnet_score": round(mvnet_score, 4),
        "policy_score": round(policy_score, 4),
        "risk_score": round(risk_score, 4),
        "decision": decision,
        "matched_rules": matched_rules
    }


# ============================================================
# TEST CASES
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MODEL 1 — RISK FUSION")
    print("=" * 70)

    test_cases = [

        {
            "name": "Normal prompt",
            "mvnet": 0.05,
            "policy": 0.00,
            "rules": []
        },

        {
            "name": "MVNet suspicious",
            "mvnet": 0.70,
            "policy": 0.00,
            "rules": []
        },

        {
            "name": "Policy suspicious",
            "mvnet": 0.30,
            "policy": 0.70,
            "rules": [
                "ignore_previous_instructions"
            ]
        },

        {
            "name": "Strong system prompt leakage",
            "mvnet": 0.20,
            "policy": 1.00,
            "rules": [
                "system_prompt_extraction"
            ]
        },

        {
            "name": "Both detectors suspicious",
            "mvnet": 0.90,
            "policy": 0.80,
            "rules": [
                "ignore_previous_instructions",
                "system_prompt_extraction"
            ]
        },

        {
            "name": "Very likely benign",
            "mvnet": 0.02,
            "policy": 0.00,
            "rules": []
        }
    ]

    for case in test_cases:

        result = calculate_risk(
            mvnet_score=case["mvnet"],
            policy_score=case["policy"],
            matched_rules=case["rules"]
        )

        print("\n")
        print("-" * 70)

        print("Test:", case["name"])

        print(
            "MVNet Score:",
            result["mvnet_score"]
        )

        print(
            "Policy Score:",
            result["policy_score"]
        )

        print(
            "Final Risk Score:",
            result["risk_score"]
        )

        print(
            "Decision:",
            result["decision"]
        )

        print(
            "Matched Rules:",
            result["matched_rules"]
        )

    print("\n")
    print("=" * 70)
    print("RISK FUSION TEST COMPLETED")
    print("=" * 70)