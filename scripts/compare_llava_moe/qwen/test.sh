function test_func() {
    echo "First line"
    echo "Second line"
    echo "Last line - this will be captured"
}

result=$(test_func)
echo "Captured in result: $result"

