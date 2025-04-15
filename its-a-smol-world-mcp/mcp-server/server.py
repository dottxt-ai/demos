from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Add a text messaging tool
@mcp.tool()
def send_text(to: str, message: str) -> str:
    """Send a text message to a contact"""
    # In a real application, this would integrate with a messaging service
    return f"Message sent to {to}: {message}"

# Add a food ordering tool
@mcp.tool()
def order_food(restaurant: str, item: str, quantity: int) -> str:
    """Order food from a restaurant"""
    # In a real application, this would integrate with a food ordering service
    return f"Ordered {quantity} {item}(s) from {restaurant}."

# Add a ride ordering tool
@mcp.tool()
def order_ride(dest: str) -> str:
    """Order a ride from a ride sharing service"""
    # In a real application, this would integrate with a ride sharing service
    return f"Ride ordered to {dest}. Your driver will arrive in 5 minutes."

# Add a weather information tool
@mcp.tool()
def get_weather(city: str) -> str:
    """Get the weather for a city"""
    # In a real application, this would integrate with a weather API
    # Using placeholder response for demo purposes
    weather_data = {
        "New York": "Partly cloudy, 72°F",
        "San Francisco": "Foggy, 58°F",
        "Los Angeles": "Sunny, 82°F",
        "Chicago": "Windy, 55°F",
        "Miami": "Rainy, 80°F"
    }
    return weather_data.get(city, f"Weather information for {city} is not available.")

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')