package ptithcm.dao;

import java.util.List;

import org.hibernate.Query;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import ptithcm.entity.LoaiSanPhamEntity;

@Repository
public class loaiSanPhamDAOIml implements loaiSanPhamDAO{
	
	@Autowired
	private SessionFactory sessionFactory;

	@Override
	public LoaiSanPhamEntity layLoaiTheoMa(String maLoai) {
		Session session = sessionFactory.getCurrentSession();
	    LoaiSanPhamEntity loai = (LoaiSanPhamEntity) session.get(LoaiSanPhamEntity.class, maLoai);
	    return loai;
	}

	@Override
	public List<LoaiSanPhamEntity> layLoai() {
		Session session = sessionFactory.getCurrentSession();
		String hql = "FROM LoaiSanPhamEntity ORDER BY tenLoai ASC";
		Query query = session.createQuery(hql);
	    List<LoaiSanPhamEntity> listLoai = query.list();
	    return listLoai;
	}

	@Override
	public List<LoaiSanPhamEntity> layLoaiTheoGioiTinh(String gioiTinh) {
		Session session = sessionFactory.getCurrentSession();
		String hql = "FROM LoaiSanPhamEntity where tenLoai LIKE :gioiTinh ";
		Query query = session.createQuery(hql).setParameter("gioiTinh", "%" +gioiTinh+"%");
	    List<LoaiSanPhamEntity> listLoai = query.list();
	    return listLoai;
	}

	@Override
	public void themLoai(LoaiSanPhamEntity loai) {
		Session session=sessionFactory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.save(loai);
			t.commit();

		} catch (Exception ex) {
			t.rollback();
			System.out.print("loi");

		} finally {
			session.close();
		}
	}

	@Override
	public void updateLoai(LoaiSanPhamEntity loai) {
		Session session=sessionFactory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.update(loai);
			t.commit();

		} catch (Exception ex) {
			t.rollback();
			System.out.print("loi");

		} finally {
			session.close();
		}
	}

	@Override
	public void xoaLoai(LoaiSanPhamEntity loai) {
		sessionFactory.getCurrentSession().delete(loai);
	}

}
