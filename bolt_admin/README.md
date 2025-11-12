# Bolt Admin UI Integration

Copy these files to your Bolt app's `app/` directory.

## File Mapping
- `layout.tsx` → `app/(admin)/layout.tsx`
- `admin/page.tsx` → `app/(admin)/admin/page.tsx`
- `admin/approval/page.tsx` → `app/(admin)/admin/approval/page.tsx`
- etc.

## Setup
1. Update ADMIN_EMAIL in layout.tsx
2. Update Modal webhook URLs in .env
3. Deploy to Bolt Cloud
