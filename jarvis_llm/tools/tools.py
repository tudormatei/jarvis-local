import ast
import re


def get_user_info(user_id, special="none"):
    return f"User {user_id} details retrieved (special: {special})"


def play_song(song_name):
    return f"🎵 Now playing: {song_name}"


tool_registry = {
    "get_user_info": get_user_info,
    "play_song": play_song,
}


def is_tool_call(response_text):
    tool_call_pattern = r"\[(\w+)\((.*?)\)\]"
    match = re.search(tool_call_pattern, response_text)
    return match


def detect_and_handle_tool_call(response_text):
    match = is_tool_call(response_text)
    if not match:
        print("❌ No tool call detected in the response.")
        return None

    func_name = match.group(1)
    params_str = match.group(2)

    try:
        fake_call = f"f({params_str})"
        parsed = ast.parse(fake_call, mode='eval')
        if not isinstance(parsed.body, ast.Call):
            raise ValueError("Not a valid function call")

        param_dict = {
            kw.arg: ast.literal_eval(kw.value)
            for kw in parsed.body.keywords
        }

    except Exception as e:
        print(f"❌ Failed to parse tool call parameters: {e}")
        return None

    print(f"\n🛠️ Tool call detected: {func_name} with params {param_dict}")

    tool_func = tool_registry.get(func_name)
    if not tool_func:
        print(f"❌ Unknown tool function: {func_name}")
        return None

    try:
        return tool_func(**param_dict)
    except Exception as e:
        print(f"❌ Error while executing '{func_name}': {e}")
        return None
