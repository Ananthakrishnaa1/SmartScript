def moderate_prompt(user_input: str) -> tuple[bool, str]:
    blocked_terms = [
        "harmful", "illegal", "inappropriate",
    ]
    
    user_input_lower = user_input.lower()
    
    for term in blocked_terms:
        if term in user_input_lower:
            return False, f"Sorry, your message contains inappropriate content ({term})"
    
    return True, ""