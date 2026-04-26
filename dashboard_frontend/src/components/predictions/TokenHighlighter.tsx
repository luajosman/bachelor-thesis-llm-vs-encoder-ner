import { Box, Chip, Stack, Typography } from "@mui/material";

import { regimeColors } from "../../app/theme";

function colorForTag(tag: string): string {
  if (tag.startsWith("B-") || tag.startsWith("I-")) {
    return regimeColors.encoder;
  }
  return "transparent";
}

export function TokenHighlighter({
  title,
  tokens,
  tags,
}: {
  title: string;
  tokens: string[];
  tags: string[];
}): JSX.Element {
  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        {title}
      </Typography>
      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        {tokens.map((token, index) => (
          <Chip
            key={`${title}-${token}-${index}`}
            label={`${token} · ${tags[index] ?? "O"}`}
            sx={{
              bgcolor: colorForTag(tags[index] ?? "O"),
              color: tags[index]?.startsWith("B-") || tags[index]?.startsWith("I-") ? "#fff" : undefined,
            }}
          />
        ))}
      </Stack>
    </Box>
  );
}
